"""Sync client: diff the music library vs the classifier JSON tree, push new
albums to the classifier server (POST /classify), store results. Idempotent
oneshot — the systemd timer just runs this; PASSENGER's auto-warm makes each
batch self-sufficient (models load in-line on a cold server).

Diff unit: the track (mtime-based, like the extractor). Batching unit: the
album (parent dir) — batches are capped by cumulative audio size so one HTTP
request stays a few GB at most. Failures are logged and retried next run.

A flock on <tree>/.sync.lock keeps concurrent syncs (timer + manual) serial.
"""
import argparse
import fcntl
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import schema


def normalize_url(u: str) -> str:
    """Accept 'host:port' as well as 'http://host:port' (urllib chokes on
    schemeless URLs: 'unknown url type')."""
    if '://' not in u:
        u = 'http://' + u
    p = urlparse(u)
    if p.scheme not in ('http', 'https') or not p.netloc:
        raise SystemExit(f'--url must be an http(s) URL, got: {u!r}')
    return u.rstrip('/')


def find_audio(music: Path):
    out = []
    for root, dirs, names in os.walk(music):
        dirs.sort()
        for name in sorted(names):
            p = Path(root) / name
            if p.suffix.lower() in schema.AUDIO_EXTS:
                out.append(p)
    return out


def pending_by_album(music: Path, tree: Path):
    """{album_dir: [(music_path, rel), ...]} for tracks needing classification."""
    albums = {}
    for p in find_audio(music):
        rel = p.relative_to(music)
        out_json = tree / rel.with_suffix('.json')
        try:
            if out_json.exists() and out_json.stat().st_mtime > p.stat().st_mtime:
                continue
        except OSError:
            pass
        albums.setdefault(rel.parent.as_posix(), []).append((p, rel.as_posix()))
    return albums


def pack_batch(albums: dict, tmpdir: Path) -> Path:
    """tar (streaming) | zstd -> temp file; returns the compressed path."""
    out = tmpdir / f'sync-batch-{int(time.time()*1000)}.tar.zst'
    z = subprocess.Popen(['zstd', '-q', '-T0', '-o', str(out)],
                         stdin=subprocess.PIPE)
    with tarfile.open(fileobj=z.stdin, mode='w|') as tf:
        for entries in albums.values():
            for path, rel in entries:
                tf.add(str(path), arcname=rel, recursive=False)
    z.stdin.close()
    if z.wait() != 0:
        raise RuntimeError('zstd failed')
    return out


def classify(url: str, batch: Path, timeout: float) -> tuple[list, list, int]:
    """POST the batch; returns (results[(rel, json_bytes)], errors, http_code)."""
    boundary_body = open(batch, 'rb')
    req = urllib.request.Request(
        url.rstrip('/') + '/classify', data=boundary_body, method='POST',
        headers={'Content-Type': 'application/x-zstd-tar',
                 'Content-Length': str(batch.stat().st_size)})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            code = resp.status
    except urllib.error.HTTPError as e:
        boundary_body.close()
        detail = e.read()[:500]
        raise RuntimeError(f'classifier returned {e.code}: {detail}') from None
    finally:
        boundary_body.close()

    results, errors = [], []
    with subprocess.Popen(['zstd', '-dc'], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE) as z:
        z.stdin.write(payload)
        z.stdin.close()
        with tarfile.open(fileobj=z.stdout, mode='r|') as tf:
            for m in tf:
                if not m.isfile():
                    continue
                data = tf.extractfile(m).read()
                if m.name == '.errors.jsonl':
                    errors = [json.loads(l) for l in data.decode().splitlines() if l.strip()]
                elif m.name.endswith('.json'):
                    results.append((m.name, data))
        if z.wait() != 0:
            raise RuntimeError('zstd decompress failed')
    return results, errors, code


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--music', required=True, help='music library root')
    ap.add_argument('--tree', required=True, help='classifier JSON tree root')
    ap.add_argument('--url', required=True, help='classifier server base URL')
    ap.add_argument('--batch-gb', type=float, default=1.5,
                    help='max original-audio size per HTTP batch')
    ap.add_argument('--timeout', type=float, default=4 * 3600,
                    help='per-batch HTTP timeout (cold server warms in-line)')
    args = ap.parse_args()

    music = Path(args.music).resolve()
    tree = Path(args.tree).resolve()
    base_url = normalize_url(args.url)
    tree.mkdir(parents=True, exist_ok=True)
    lockfile = open(tree / '.sync.lock', 'w')
    try:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit('another sync is already running (.sync.lock held)')

    albums = pending_by_album(music, tree)
    n_tracks = sum(len(v) for v in albums.values())
    print(f'{n_tracks} tracks in {len(albums)} albums pending', flush=True)
    if not albums:
        print('nothing to do', flush=True)
        return

    # album -> batch assignment by cumulative audio size (sorted for stability)
    batches = []
    cur, cur_size = {}, 0
    for name in sorted(albums):
        size = sum(p.stat().st_size for p, _ in albums[name])
        if cur and cur_size + size > args.batch_gb * 1e9:
            batches.append(cur)
            cur, cur_size = {}, 0
        cur[name] = albums[name]
        cur_size += size
    if cur:
        batches.append(cur)
    print(f'{len(batches)} batch(es) (cap {args.batch_gb}GB)', flush=True)

    t0 = time.time()
    total_ok = total_err = 0
    for i, batch in enumerate(batches, 1):
        batch_tracks = sum(len(v) for v in batch.values())
        print(f'batch {i}/{len(batches)}: {batch_tracks} tracks, '
              f'{len(batch)} albums — packing...', flush=True)
        packed = pack_batch(batch, tree)
        try:
            print(f'batch {i}/{len(batches)}: {packed.stat().st_size/1e6:.0f}MB '
                  f'compressed — sending to {base_url}...', flush=True)
            results, errors, _ = classify(base_url, packed, args.timeout)
        finally:
            packed.unlink(missing_ok=True)
        for rel, data in results:
            out_json = tree / rel
            out_json.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_json.with_suffix('.json.tmp')
            with open(tmp, 'wb') as f:
                f.write(data)
            tmp.rename(out_json)
        errlog = tree / '.errors.jsonl'
        if errors:
            with open(errlog, 'a') as f:
                for e in errors:
                    f.write(json.dumps(e) + '\n')
        total_ok += len(results)
        total_err += len(errors)
        print(f'batch {i}/{len(batches)}: stored {len(results)}, '
              f'errors {len(errors)} (total {total_ok} ok / {total_err} err, '
              f'{time.time()-t0:.0f}s elapsed)', flush=True)

    print(f'SYNC DONE: {total_ok} stored, {total_err} errors, {time.time()-t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    main()
