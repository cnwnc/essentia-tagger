"""Blob server: GET/HEAD for the similarity blob with ETag/Last-Modified so
the KAZOOIE proxy can revalidate with conditional GETs (304 = no re-download).
stdlib http.server; no auth — bind to the wireguard interface (wg = trust
boundary). Single file, so ETag = size-mtime and reads are mmap-free.
"""
import argparse
import email.utils
import http.server
import os
import sys
import threading
from pathlib import Path

# client-disconnect errors: normal (e.g. ^C'd curl mid-download), never traceback
DISCONNECT = (BrokenPipeError, ConnectionResetError, TimeoutError)


class FileState:
    """Stat-derived ETag state for one file."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.refresh()

    def refresh(self):
        with self.lock:
            st = self.path.stat()
            self.size = st.st_size
            self.mtime = int(st.st_mtime)
            self.etag = f'"{self.size:x}-{self.mtime:x}"'
            self.last_modified = email.utils.formatdate(self.mtime, usegmt=True)


class AtlasStore:
    """atlas-<n>.webp files in a dir, ETag per file. Name-sanitizing."""

    def __init__(self, atlas_dir: Path):
        self.dir = Path(atlas_dir)
        self.lock = threading.Lock()
        self.files = {}

    def get(self, name):
        if not (name.startswith('atlas-') and name.endswith('.webp') and
                name[len('atlas-'):-len('.webp')].isdigit()):
            return None
        with self.lock:
            st = self.files.get(name)
            try:
                if st is None:
                    st = FileState(self.dir / name)
                    self.files[name] = st
                else:
                    st.refresh()
                if not st.path.exists():
                    return None
                return st
            except OSError:
                self.files.pop(name, None)
                return None


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    blob: FileState = None
    atlas: AtlasStore = None  # set by serve() when --atlas-dir is given

    def _send_headers(self, st: FileState, code, length=None):
        self.send_response(code)
        self.send_header('ETag', st.etag)
        self.send_header('Last-Modified', st.last_modified)
        self.send_header('Cache-Control', 'public, max-age=300')
        self.send_header('Accept-Ranges', 'none')
        # wg-only listener: origin restrictions buy nothing, and the dev
        # frontend loads these artifacts cross-origin (file:// / localhost)
        self.send_header('Access-Control-Allow-Origin', '*')
        if length is not None:
            self.send_header('Content-Length', str(length))
        self.end_headers()

    def _handle_file(self, st: FileState, head_only: bool):
        inm = self.headers.get('If-None-Match')
        ims = self.headers.get('If-Modified-Since')
        not_modified = (inm and inm.strip() == st.etag) or \
                       (inm is None and ims and st.last_modified == ims)
        try:
            if not_modified:
                self._send_headers(st, 304, 0)
                return
            self._send_headers(st, 200, st.size)
            if head_only:
                return
            with open(st.path, 'rb') as f:
                while chunk := f.read(1 << 20):
                    self.wfile.write(chunk)
        except DISCONNECT:
            self.close_connection = True  # client vanished mid-response

    def do_GET(self):
        if self.path in ('/', '/blob', '/albums'):
            self._handle_file(self.blob, head_only=False)
        elif self.path.startswith('/atlas/') and self.atlas is not None:
            st = self.atlas.get(self.path[len('/atlas/'):])
            if st is None:
                self._send_headers(self.blob, 404, 0)
            else:
                self._handle_file(st, head_only=False)
        else:
            self._send_headers(self.blob, 404, 0)

    def do_HEAD(self):
        if self.path in ('/', '/blob', '/albums'):
            self._handle_file(self.blob, head_only=True)
        elif self.path.startswith('/atlas/') and self.atlas is not None:
            st = self.atlas.get(self.path[len('/atlas/'):])
            if st is None:
                self._send_headers(self.blob, 404, 0)
            else:
                self._handle_file(st, head_only=True)
        else:
            self._send_headers(self.blob, 404, 0)

    def log_message(self, fmt, *args):
        print(f'{self.address_string()} {fmt % args}', flush=True)


class QuietServer(http.server.ThreadingHTTPServer):
    """Silences the per-connection traceback noise for disconnects."""

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], DISCONNECT):
            return
        super().handle_error(request, client_address)


def serve(blob_path, bind, port, atlas_dir=None):
    Handler.blob = FileState(blob_path)
    if atlas_dir:
        Handler.atlas = AtlasStore(Path(atlas_dir))
    QuietServer((bind, port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--blob', required=True, help='blob file to serve')
    ap.add_argument('--atlas-dir', default=None,
                    help='dir with atlas-<n>.webp sheets (enables /atlas/<n>.webp)')
    ap.add_argument('--bind', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=9478)
    args = ap.parse_args()
    print(f'serving {args.blob}' +
          (f' + atlas {args.atlas_dir}' if args.atlas_dir else '') +
          f' on {args.bind}:{args.port}', flush=True)
    serve(args.blob, args.bind, args.port, args.atlas_dir)


if __name__ == '__main__':
    main()
