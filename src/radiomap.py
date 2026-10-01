"""Radio map builder: id_hash -> MusicBrainz release id (Contract G sidecar).

The blob's per-album id_hash is sha256(first-track-artist / album-dir)[:8] LE,
a filesystem-identity hash navidrome cannot answer at runtime. Navidrome DOES
expose musicBrainzId per album (Subsonic search3 / album lists), and our
library tree records the canonical MB release per album dir in album.yaml.
This tool joins the two worlds into a flat JSON map consumed server-side by
the site backend to resolve a blob album to its navidrome album.

  radiomap.py --blob <blob(.gz)> --music <mount> --out radio-map.json

Coverage note: albums whose album.yaml lacks musicbrainz.release are omitted
here; consumers fall back to artist+album name search (same strategy as the
atlas builder's fallback path).
"""

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blob as blobmod

RELEASE_RE = re.compile(r'^(\s+)release: ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}'
                        r'-[0-9a-f]{4}-[0-9a-f]{12})\s*$', re.M)
ARTIST_RE = re.compile(r'^(\s+)artist: (.+)$', re.M)


def read_release_and_artist(ypath: Path):
    try:
        import yaml
        with open(ypath) as f:
            mb = (yaml.safe_load(f) or {}).get('musicbrainz') or {}
            tracks = (yaml.safe_load(f) or {}).get('tracks') or []
        release = mb.get('release')
        artist = (tracks[0] or {}).get('artist') if tracks else None
        return release, artist
    except ImportError:
        pass
    txt = ypath.read_text(errors='replace')
    head = txt.split('tracks:', 1)[0]
    rel = RELEASE_RE.search(txt)
    art = ARTIST_RE.search(head)
    return (rel.group(2) if rel else None,
            art.group(2).strip() if art else None)


def blob_albums(path: Path):
    raw = path.read_bytes()
    if raw[:2] == b'\x1f\x8b':
        raw = zlib.decompress(raw, 31)
    n = struct.unpack_from('<I', raw, 20)[0]
    ncls = struct.unpack_from('<I', raw, 24)[0]
    p = 32
    for _ in range(ncls):
        (sl,) = struct.unpack_from('<I', raw, p)
        p += 4 + sl
    out = []
    for _ in range(n):
        p += 4
        (al,) = struct.unpack_from('<I', raw, p)
        artist = raw[p + 4:p + 4 + al].decode('utf-8')
        p += 4 + al
        (al,) = struct.unpack_from('<I', raw, p)
        album = raw[p + 4:p + 4 + al].decode('utf-8')
        p += 4 + al + 4 + 4
        (ih,) = struct.unpack_from('<Q', raw, p)
        p += 8
        p += ncls * 2
        out.append((ih, artist, album))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--blob', required=True, type=Path)
    ap.add_argument('--music', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()

    dirs = {}
    for artist_dir in sorted(os.listdir(args.music)):
        base = args.music / artist_dir
        if not base.is_dir():
            continue
        for alb in sorted(os.listdir(base)):
            ypath = base / alb / 'album.yaml'
            if ypath.is_file():
                dirs[f'{artist_dir}/{alb}'] = ypath

    albums = blob_albums(args.blob)
    mapping = {}
    misses = []
    for ih, artist, album in albums:
        rel_dir = next((d for d in dirs if d.split('/', 1)[1] == album
                        and dirs[d].parent.parent.name == artist), None)
        if rel_dir is None:
            misses.append((artist, album))
            continue
        release, first_artist = read_release_and_artist(dirs[rel_dir])
        if release is None:
            misses.append((artist, album))
            continue
        key_artist = first_artist or artist
        expected = blobmod.album_id_hash(key_artist, rel_dir)
        if expected != ih:
            misses.append((artist, album))
            continue
        mapping[str(ih)] = release

    args.out.write_text(json.dumps(mapping, indent=0) + '\n')
    print(f'radio map: {len(mapping)} entries, {len(misses)} unresolved')
    for artist, album in misses[:10]:
        print(f'  unresolved: {artist} --- {album}', file=sys.stderr)


if __name__ == '__main__':
    main()
