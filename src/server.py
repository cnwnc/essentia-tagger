"""Classifier server: POST /classify — zstd tar of audio in, zstd tar of
per-track JSONs out (Contract A wire format). The ONLY functional route;
GET /health is liveness/monitoring only.

Semantics:
  - stateless: request tarball stages in /dev/shm, deleted after response
  - auto-warm: a cold request loads models in-line (~15-30s) then processes;
    after --idle-timeout (default 600s) with no requests the models unload
    (idle cost ~0) and the next request re-warms transparently
  - single job slot: concurrent POST -> 409
  - per-track errors are in-band (.errors.jsonl in the response tar);
    4xx/5xx only for whole-batch failure (bad tar, missing models)
  - no auth: bind to the wireguard IP (wg admission is the trust boundary)
"""
import argparse
import gc
import http.server
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from pathlib import Path

import core
import schema

DISCONNECT = (BrokenPipeError, ConnectionResetError, TimeoutError)


class Engine:
    """Owns the Classifier; lazy init, idle unload, single-job lock."""

    def __init__(self, idle_timeout: float):
        self.idle_timeout = idle_timeout
        self._clf = None
        self._last_used = 0.0
        self._lock = threading.Lock()   # classifies this engine (job slot)
        self._state_lock = threading.Lock()

    @property
    def warm(self):
        return self._clf is not None

    def acquire(self, blocking: bool):
        return self._lock.acquire(blocking=blocking)

    def release(self):
        self._lock.release()

    def classifier(self):
        with self._state_lock:
            if self._clf is None:
                print('warm: loading models...', flush=True)
                t0 = time.time()
                self._clf = core.Classifier()
                print(f'warm: models ready in {time.time()-t0:.0f}s', flush=True)
            self._last_used = time.time()
            return self._clf

    def idle_watchdog(self):
        while True:
            time.sleep(30)
            with self._state_lock:
                if self._clf is not None and \
                        time.time() - self._last_used > self.idle_timeout:
                    print('idle: unloading models', flush=True)
                    self._clf = None
                    gc.collect()


def safe_extract(tar_path: Path, dest: Path):
    """Extract zstd|tar with path-traversal protection (py3.10 has no filter=)."""
    with subprocess.Popen(['zstd', '-dc', str(tar_path)], stdout=subprocess.PIPE) as z:
        with tarfile.open(fileobj=z.stdout, mode='r|') as tf:
            for m in tf:
                if not m.isfile():
                    continue
                name = Path(m.name)
                if name.is_absolute() or '..' in name.parts:
                    raise ValueError(f'unsafe tar member: {m.name}')
                tf.extract(m, dest)


def pack_response(results: list, errors: list) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tf:
        for rel, data in results:
            # tree convention: audio ext replaced by .json (out/<Artist>/.../NN Title.json)
            name = str(Path(rel).with_suffix('.json'))
            b = json.dumps(data, separators=(',', ':')).encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(b)
            info.mtime = int(time.time())
            tf.addfile(info, io.BytesIO(b))
        if errors:
            eb = '\n'.join(json.dumps(e) for e in errors).encode()
            info = tarfile.TarInfo(name='.errors.jsonl')
            info.size = len(eb)
            info.mtime = int(time.time())
            tf.addfile(info, io.BytesIO(eb))
    raw = buf.getvalue()
    z = subprocess.run(['zstd', '-q', '-T0'], input=raw, capture_output=True, check=True)
    return z.stdout


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    engine: Engine = None
    shm_dir: Path = None

    # ---- helpers -------------------------------------------------------
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        try:
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except DISCONNECT:
            self.close_connection = True

    def do_GET(self):
        if self.path == '/health':
            self._json(200, {'ok': True, 'warm': self.engine.warm,
                             'model': core.EMB_MODEL_NAME,
                             'head': core.HEAD_MODEL_NAME,
                             'schema': schema.SCHEMA_VERSION})
        else:
            self._json(404, {'error': 'not found'})

    # ---- the job -------------------------------------------------------
    def do_POST(self):
        if self.path != '/classify':
            self._json(404, {'error': 'not found'})
            return
        if not self.engine.acquire(blocking=False):
            self._json(409, {'error': 'busy: another classification job is running'})
            return
        try:
            self._classify()
        finally:
            self.engine.release()

    def _classify(self):
        job = f'tagger-{uuid.uuid4().hex}'
        stage = self.shm_dir / job
        tar_in = Path('/dev/shm') / f'{job}.tar.zst'
        try:
            # 1. read the request body to /dev/shm
            length = int(self.headers.get('Content-Length', 0))
            if length <= 0:
                self._json(400, {'error': 'Content-Length required'})
                return
            t0 = time.time()
            written = 0
            with open(tar_in, 'wb') as f:
                while written < length:
                    chunk = self.rfile.read(min(1 << 20, length - written))
                    if not chunk:
                        raise ConnectionError('client disconnected mid-upload')
                    f.write(chunk)
                    written += len(chunk)
            print(f'job {job[:12]}: {written/1e6:.0f}MB received in {time.time()-t0:.0f}s',
                  flush=True)

            # 2. extract
            stage.mkdir(parents=True)
            try:
                safe_extract(tar_in, stage)
            except Exception as e:
                self._json(400, {'error': f'bad tarball: {e}'})
                return
            os.remove(tar_in)

            # 3. classify every file (in-band per-track errors)
            clf = self.engine.classifier()
            results, errors = [], []
            files = sorted(p for p in stage.rglob('*')
                           if p.is_file() and p.suffix.lower() in schema.AUDIO_EXTS)
            t0 = time.time()
            for p in files:
                rel = p.relative_to(stage).as_posix()
                try:
                    from essentia.standard import MonoLoader
                    audio = MonoLoader(filename=str(p),
                                       sampleRate=schema.SAMPLE_RATE)()
                    results.append((rel, core.process_track(clf, rel, audio)))
                except Exception as e:
                    msg = f'{type(e).__name__}: {e}'
                    print(f'job {job[:12]}: ERROR {rel}: {msg}', file=sys.stderr, flush=True)
                    errors.append({'path': rel, 'error': msg})
            print(f'job {job[:12]}: {len(results)} tracks classified '
                  f'({len(errors)} errors) in {time.time()-t0:.0f}s', flush=True)

            # 4. respond
            body = pack_response(results, errors)
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-zstd-tar')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except DISCONNECT:
                self.close_connection = True
                print(f'job {job[:12]}: client disconnected before response completed',
                      file=sys.stderr, flush=True)
        except Exception as e:
            try:
                self._json(500, {'error': f'{type(e).__name__}: {e}'})
            except Exception:
                pass
            print(f'job {job[:12]}: FAILED: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            tar_in.unlink(missing_ok=True)

    def log_message(self, fmt, *args):
        print(f'{self.address_string()} {fmt % args}', flush=True)


class QuietServer(http.server.ThreadingHTTPServer):
    """Silences the per-connection traceback noise for disconnects."""

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], DISCONNECT):
            return
        super().handle_error(request, client_address)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--bind', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=9478)
    ap.add_argument('--idle-timeout', type=float, default=600,
                    help='seconds of inactivity before unloading models')
    args = ap.parse_args()

    Handler.engine = Engine(args.idle_timeout)
    Handler.shm_dir = Path('/dev/shm')
    threading.Thread(target=Handler.engine.idle_watchdog, daemon=True).start()
    print(f'classifier listening on {args.bind}:{args.port} '
          f'(idle timeout {args.idle_timeout:.0f}s)', flush=True)
    QuietServer((args.bind, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
