"""Bulk local-tree backfill CLI: mirror a music tree to one JSON per track.

The steady-state path is server.py + sync.py; this tool is for one-time
backfills and manual top-ups (untagged-album detection = mtime resume below,
so re-running it after any interruption finishes exactly what's missing).

Resume: skips tracks whose output json is newer than the audio file.
Errors: in-band to OUT/.errors.jsonl, non-fatal (--strict to abort).
Decode: essentia MonoLoader; ffmpeg fallback for codecs it lacks (opus).
Performance: spawn Pool of loader processes — essentia's python calls hold
the GIL, so threads don't overlap; sshfs decode is the bottleneck.
"""
import argparse
import base64
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

import schema

_MOUNT = None  # set in loader processes via initializer


def find_audio_files(mount: Path):
    files = []
    for root, dirs, names in os.walk(mount):
        dirs.sort()
        for name in sorted(names):
            p = Path(root) / name
            if p.suffix.lower() in schema.AUDIO_EXTS:
                files.append(p)
    return files


def _pool_init(mount_str):
    global _MOUNT
    _MOUNT = mount_str


def load_task(rel_str):
    """Runs in loader processes: decode audio, return it (or an error string)."""
    path = Path(_MOUNT) / rel_str
    try:
        from essentia.standard import MonoLoader
        audio = MonoLoader(filename=str(path), sampleRate=schema.SAMPLE_RATE)()
        return (rel_str, None, audio)
    except Exception as mono_err:
        # this essentia build lacks some codecs (e.g. opus): fall back to ffmpeg
        try:
            import subprocess
            proc = subprocess.run(
                ['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'f32le',
                 '-ac', '1', '-ar', str(schema.SAMPLE_RATE), '-'],
                capture_output=True, check=True)
            audio = np.frombuffer(proc.stdout, dtype=np.float32)
            if audio.size == 0:
                raise RuntimeError('ffmpeg produced no audio')
            return (rel_str, None, audio)
        except Exception:
            return (rel_str, f'{type(mono_err).__name__}: {mono_err}', None)


def process_track(classifier, rel: Path, audio: np.ndarray):
    """Deprecated shim — moved to core.process_track (shared with server.py)."""
    return core.process_track(classifier, rel, audio)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--mount', default='MOUNT', type=Path)
    ap.add_argument('--out', default='out', type=Path)
    ap.add_argument('--limit', type=int, default=0, help='process at most N tracks (0 = all)')
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--artist', action='append', default=[],
                    help='only process these artist dirs (repeatable)')
    ap.add_argument('--strict', action='store_true', help='abort on first error')
    args = ap.parse_args()

    mount = args.mount.resolve()
    out = args.out.resolve()
    errlog = out / '.errors.jsonl'
    out.mkdir(parents=True, exist_ok=True)

    import core
    classifier = core.Classifier()

    files = find_audio_files(mount)
    total_all = len(files)
    print(f'{total_all} audio files found under {mount}', flush=True)

    tasks = []
    skipped = 0
    t_scan = time.time()
    for path in files:
        rel = path.relative_to(mount)
        if args.artist and rel.parts[0] not in args.artist:
            continue
        out_json = out / rel.with_suffix('.json')
        try:
            if out_json.exists() and out_json.stat().st_mtime > path.stat().st_mtime:
                skipped += 1
                continue
        except OSError:
            pass  # try to process anyway
        tasks.append(str(rel))
        if args.limit and len(tasks) >= args.limit:
            break
    total = len(tasks)
    print(f'{total} to process ({skipped} already done, scan {time.time()-t_scan:.0f}s)', flush=True)

    done = errors = 0
    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=args.workers, initializer=_pool_init,
                  initargs=(str(mount),)) as pool:
        for rel_str, err, audio in pool.imap(load_task, tasks, chunksize=1):
            rel = Path(rel_str)
            if err is not None:
                errors += 1
                print(f'ERROR: {rel}: {err}', file=sys.stderr)
                with open(errlog, 'a') as f:
                    f.write(json.dumps({'path': rel_str, 'error': err, 'ts': time.time()}) + '\n')
                if args.strict:
                    pool.terminate()
                    raise RuntimeError(err)
                continue
            try:
                result = process_track(classifier, rel, np.asarray(audio))
                out_json = out / rel.with_suffix('.json')
                out_json.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_json.with_suffix('.json.tmp')
                with open(tmp, 'w') as f:
                    json.dump(result, f, separators=(',', ':'))
                tmp.rename(out_json)
                done += 1
            except Exception as e:
                errors += 1
                print(f'ERROR: {rel}: {type(e).__name__}: {e}', file=sys.stderr)
                with open(errlog, 'a') as f:
                    f.write(json.dumps({'path': rel_str, 'error': f'{type(e).__name__}: {e}',
                                        'ts': time.time()}) + '\n')
                if args.strict:
                    pool.terminate()
                    raise

            if done % 25 == 0:
                dt = time.time() - t0
                rate = done / max(dt, 0.01)
                eta_min = (total - done) / max(rate, 0.01) / 60
                print(f'[{done + skipped}/{total_all}] done={done} err={errors} '
                      f'rate={rate:.2f}/s eta={eta_min:.0f}min', flush=True)

    dt = time.time() - t0
    print(f'FINISHED in {dt:.0f}s: processed={done} skipped={skipped} errors={errors}', flush=True)


if __name__ == '__main__':
    main()
