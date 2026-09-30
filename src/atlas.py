"""Atlas builder: cover-art sprite sheets + tile manifest (Contract F).

Per album in the classifier tree:
  1. album.yaml from the MUSIC tree (same dir as tracks) → musicbrainz.release
  2. navidrome search3 → album whose musicBrainzId matches (EXACT; fallback:
     artist+album string search with MBID cross-check)
  3. getCoverArt?size=128 (ND's server-side thumbnail; bytes may be PNG/JPEG/anything)
  4. ffmpeg → 128x128 lossy WebP tile (input-format-agnostic; corrupt input =
     fallback tile: solid color from the album's dominant activation parent)
  5. pack tiles row-major into sheet_dim x sheet_dim sheets → atlas-N.webp

Outputs (in --out-dir):
  atlas-N.webp                 assembled sheet(s), lossy webp q80
  atlas-manifest.json          {schema, sheet_dim, tile, sheets, sha256, albums{dir→(sheet,tx,ty)}}
  cache/<key>.webp             per-album tile cache (key = sha1(album_dir | coverArt id))

Incremental: cached tiles are reused without network; only new/changed albums
hit ND (cache key includes ND's coverArt id, which embeds a content hash).
Artless albums get a generated fallback tile so every album has a uniform
tile in the atlas — no special casing downstream.
"""
import argparse
import colorsys
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

import blob as blobmod
import schema

TILE = 128
SHEET_DIM = 8192
PER_ROW = SHEET_DIM // TILE  # 64 tiles per row/col → 4096 per sheet
ND_TIMEOUT = 30


class ND:
    """Minimal Subsonic/OpenSubsonic client (u/p auth, trailing-slash safe)."""

    def __init__(self, base, user, password):
        self.base = base.rstrip('/') + '/'
        self.auth = {'u': user, 'p': password, 'v': '1.8.0', 'c': 'tagger-atlas',
                     'f': 'json'}

    def _url(self, endpoint, **params):
        q = urllib.parse.urlencode({**self.auth, **params})
        return self.base + f'rest/{endpoint}?' + q

    def _get_json(self, endpoint, **params):
        with urllib.request.urlopen(self._url(endpoint, **params), timeout=ND_TIMEOUT) as r:
            d = json.load(r)
        resp = d.get('subsonic-response', {})
        if resp.get('status') != 'ok':
            raise RuntimeError(f'{endpoint}: {resp}')
        return resp

    def find_album(self, artist, album, mbid=None):
        """search3 → album dict; exact MBID match preferred over name match."""
        for query in filter(None, [f'{artist} {album}', album]):
            resp = self._get_json('search3', query=query, artistCount=0,
                                  albumCount=10, songCount=0)
            hits = resp.get('searchResult3', {}).get('album', [])
            for h in hits:
                if mbid and h.get('musicBrainzId') == mbid:
                    return h
            if mbid:
                continue  # with an MBID, only an exact match counts
            for h in hits:
                if h.get('name', '').strip().lower() == album.strip().lower():
                    return h
        return None

    def cover_bytes(self, album):
        cid = album.get('coverArt') or f'al-{album["id"]}'
        req = urllib.request.Request(
            self._url('getCoverArt', id=cid, size=TILE))
        with urllib.request.urlopen(req, timeout=ND_TIMEOUT) as r:
            data = r.read()
        if b'<html' in data[:200] or not data:
            raise RuntimeError('cover request returned no image data')
        return data


def read_album_yaml(path: Path):
    """Minimal yaml reader for the album.yaml subset (no external deps)."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        pass
    # crude fallback: top-level scalars + musicbrainz block (enough for our keys)
    out, in_mb = {}, False
    with open(path, errors='replace') as f:
        for line in f:
            if not line.strip() or line.strip().startswith('#'):
                continue
            if line.startswith(('tracks:', 'provenance:', 'tags:')):
                in_mb = False
                continue
            if line.startswith('musicbrainz:'):
                in_mb = True
                continue
            if in_mb and line.startswith('    '):
                k, _, v = line.strip().partition(':')
                if k == 'release':
                    out['musicbrainz'] = {'release': v.strip().strip('"\'')}
                continue
            if line.startswith('  '):
                continue
            k, _, v = line.partition(':')
            out[k.strip()] = v.strip().strip('"\'')
    return out


def dominant_parent_color(track_files, classes):
    """Solid fallback color: hue from the dominant activation parent."""
    import json as _json
    parents = {}
    for tf in track_files:
        with open(tf) as f:
            d = _json.load(f)
        for k, v in d.get('activations', {}).items():
            p = k.split('---', 1)[0]
            parents[p] = parents.get(p, 0.0) + float(v)
    if not parents:
        hue = 0.0
    else:
        top = max(parents, key=parents.get)
        hue = (hash(top) % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.52, 0.42)
    return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


def tile_to_rgb(webp_bytes: bytes) -> np.ndarray:
    """Any-format image bytes (webp here) → exact (TILE, TILE, 3) rgb array."""
    p = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', '-',
         '-vf', f'scale={TILE}:{TILE}:force_original_aspect_ratio=increase,'
                f'crop={TILE}:{TILE}',
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        input=webp_bytes, capture_output=True, check=True)
    arr = np.frombuffer(p.stdout, dtype=np.uint8)
    if arr.size != TILE * TILE * 3:
        raise RuntimeError(f'unexpected raw size {arr.size}')
    return arr.reshape(TILE, TILE, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--tree', required=True, help='classifier JSON tree (album set source)')
    ap.add_argument('--music', required=True, help='music library root (album.yaml source)')
    ap.add_argument('--out-dir', required=True, help='atlas output dir')
    ap.add_argument('--nd-base', default=os.environ.get('NAVIDROME_BASE_URL', ''))
    ap.add_argument('--nd-user', default=os.environ.get('NAVIDROME_USERNAME', ''))
    ap.add_argument('--nd-pass', default=os.environ.get('NAVIDROME_PASSWORD', ''))
    ap.add_argument('--limit', type=int, default=0, help='only first N albums (testing)')
    args = ap.parse_args()
    if not (args.nd_base and args.nd_user and args.nd_pass):
        raise SystemExit('navidrome creds required (--nd-* or NAVIDROME_* env)')

    import core
    _, _, classes_json = core.model_paths()
    classes = core.load_classes(classes_json)
    tree = Path(args.tree).resolve()
    music = Path(args.music).resolve()
    out_dir = Path(args.out_dir).resolve()
    cache = out_dir / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    nd = ND(args.nd_base, args.nd_user, args.nd_pass)

    # album set: same grouping as build.py (parent dir of each track json)
    albums = {}
    for root, dirs, names in os.walk(tree):
        dirs.sort()
        for name in sorted(names):
            if name.endswith('.json') and not name.startswith('.'):
                p = Path(root) / name
                albums.setdefault(p.relative_to(tree).parent.as_posix(), []).append(p)
    album_dirs = sorted(albums)
    if args.limit:
        album_dirs = album_dirs[:args.limit]
    n = len(album_dirs)
    print(f'{n} albums to atlas (tree={tree})', flush=True)

    manifest_path = out_dir / 'atlas-manifest.json'
    manifest = {'schema': 1, 'tile': TILE, 'sheet_dim': SHEET_DIM,
                'sheets': [], 'sha256': [], 'albums': {}, 'updated_ts': 0}
    if manifest_path.exists():
        old = json.load(open(manifest_path))
        if old.get('tile') == TILE and old.get('sheet_dim') == SHEET_DIM:
            manifest = old  # keep existing assignments for incremental reuse
    assignments = {}

    # pass 1: resolve every album to tile rgb data (network for new only)
    tiles_rgb = []
    misses = 0
    t0 = time.time()
    for i, d in enumerate(album_dirs):
        key_src = None
        tile_bytes = None
        mb = {}
        ypath = music / d / 'album.yaml'
        if ypath.exists():
            mb = read_album_yaml(ypath) or {}
        artist = mb.get('albumartist') or d.split('/')[0]
        album = mb.get('album') or d.split('/')[-1]
        release = (mb.get('musicbrainz') or {}).get('release')
        # existing assignment with a live cache file? skip network entirely
        prev = manifest['albums'].get(d)
        if prev and prev.get('cover_key'):
            cfile = cache / f"{prev['cover_key']}.webp"
            if cfile.exists():
                tiles_rgb.append(tile_to_rgb(cfile.read_bytes()))
                assignments[d] = {'sheet': prev['sheet'], 'tx': prev['tx'],
                                  'ty': prev['ty'], 'cover_key': prev['cover_key']}
                continue
        try:
            hit = nd.find_album(artist, album, mbid=release)
            if hit is None:
                raise RuntimeError(f'no ND match for {d}')
            key = hashlib.sha1(
                f'{d}|{hit.get("coverArt", hit["id"])}'.encode()).hexdigest()[:20]
            cfile = cache / f'{key}.webp'
            if cfile.exists():
                # cached from a previous (possibly interrupted) run
                tiles_rgb.append(tile_to_rgb(cfile.read_bytes()))
            else:
                raw = nd.cover_bytes(hit)
                tile_bytes = tile_to_rgb(raw).tobytes()
                cfile.write_bytes(
                    subprocess.run(
                        ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                         '-s', f'{TILE}x{TILE}', '-r', '1', '-i', '-',
                         '-c:v', 'libwebp', '-q:v', '80', '-frames:v', '1',
                         '-f', 'webp', '-'],
                        input=tile_bytes, capture_output=True, check=True).stdout)
                tiles_rgb.append(np.frombuffer(tile_bytes, dtype=np.uint8).reshape(TILE, TILE, 3))
            assignments[d] = {'cover_key': key}  # sheet/tx/ty assigned at packing
        except Exception as e:
            misses += 1
            if misses <= 5:
                print(f'  fallback tile for {d}: {e}', file=sys.stderr, flush=True)
            color = dominant_parent_color(albums[d], classes)
            tile = np.tile(color, (TILE, TILE, 1))
            tiles_rgb.append(tile)
            assignments[d] = {'cover_key': None}
        if (i + 1) % 100 == 0:
            print(f'  [{i + 1}/{n}] resolved ({misses} fallbacks, '
                  f'{time.time()-t0:.0f}s)', flush=True)

    # pass 2: pack into sheets — keep old positions when still valid, fill gaps
    retained = {d: a for d, a in assignments.items() if 'sheet' in a}
    used = {(a['sheet'], a['tx'], a['ty']) for a in retained.values()}
    max_sheet = max((a['sheet'] for a in retained.values()), default=-1)
    free = [(s, tx, ty) for s in range(max_sheet + 1)
            for ty in range(PER_ROW) for tx in range(PER_ROW)
            if (s, tx, ty) not in used]
    n_new = n - len(retained)
    while len(free) < n_new:
        max_sheet += 1
        free += [(max_sheet, tx, ty) for ty in range(PER_ROW)
                 for tx in range(PER_ROW)]
    for d, a in assignments.items():
        if 'sheet' in a:
            continue
        s, tx, ty = free.pop(0)
        a['sheet'], a['tx'], a['ty'] = s, tx, ty
    n_sheets = max_sheet + 1

    # pass 3: assemble + encode sheets
    manifest['sheets'], manifest['sha256'], manifest['albums'] = [], [], {}
    for s in range(n_sheets):
        sheet = np.zeros((SHEET_DIM, SHEET_DIM, 3), dtype=np.uint8)
        for d, a in assignments.items():
            if a['sheet'] != s:
                continue
            idx = album_dirs.index(d)
            sheet[a['ty'] * TILE:(a['ty'] + 1) * TILE,
                  a['tx'] * TILE:(a['tx'] + 1) * TILE] = tiles_rgb[idx]
            manifest['albums'][d] = {'sheet': s, 'tx': a['tx'], 'ty': a['ty'],
                                     'cover_key': a.get('cover_key')}
        name = f'atlas-{s}.webp'
        p = subprocess.run(
            ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
             '-s', f'{SHEET_DIM}x{SHEET_DIM}', '-r', '1', '-i', '-',
             '-c:v', 'libwebp', '-q:v', '80', '-frames:v', '1', str(out_dir / name)],
            input=sheet.tobytes(), capture_output=True, check=True)
        digest = hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
        manifest['sheets'].append(name)
        manifest['sha256'].append(digest)
        print(f'  sheet {s}: {name} ({(out_dir / name).stat().st_size/1e6:.1f} MB, '
              f'sha256 {digest[:12]}...)', flush=True)
    manifest['updated_ts'] = int(time.time())
    tmp = manifest_path.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(manifest, f)
    tmp.rename(manifest_path)

    have = sum(1 for a in manifest['albums'].values() if True)
    print(f'ATLAS DONE: {n_sheets} sheet(s), {have}/{n} albums, '
          f'{misses} fallback tiles, {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
