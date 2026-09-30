"""Blob builder: classifier JSON tree -> similarity blob (Contract D).

Pure numpy; cheap to re-run (weights/aggregation are rebuild knobs,
re-extraction is never needed). Steps:
  1. scan tree, load track JSONs, group by parent dir (= album unit)
  2. album profile = element-wise MEDIAN over tracks (activations + embedding)
  3. parent shares = row-normalized activations (sum -> 1 per album)
  4. per-channel cosine matrices (raw, stored as-is for the frontend)
  5. blend = w_act * rank(act) + w_emb * rank(emb) + artist-bonus
     (rank = off-diagonal percentile; embeddings are rank-only anyway)
  6. top-k neighbors by blend
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import blob as blobmod
import core
import schema


def scan_tracks(tree: Path):
    """(json_path, rel_posix) for every track JSON in the tree."""
    for root, dirs, names in os.walk(tree):
        dirs.sort()
        for name in sorted(names):
            if name.endswith('.json') and not name.startswith('.'):
                p = Path(root) / name
                rel = p.relative_to(tree).as_posix()
                yield p, rel


def load_track(p: Path):
    with open(p) as f:
        return json.load(f)


def rank_scale(m: np.ndarray) -> np.ndarray:
    """Off-diagonal percentile scaling in [0,1]; diagonal -> 1."""
    n = m.shape[0]
    iu = np.triu_indices(n, 1)
    vals = m[iu]
    order = np.argsort(vals, kind='stable')
    pct = np.empty(vals.shape, dtype=np.float32)
    pct[order] = np.arange(vals.size, dtype=np.float32) / max(vals.size - 1, 1)
    out = np.zeros((n, n), dtype=np.float32)
    out[iu] = pct
    out = out + out.T
    np.fill_diagonal(out, 1.0)
    return out


def cosines(x: np.ndarray) -> np.ndarray:
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return (xn @ xn.T).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--tree', required=True, help='classifier JSON tree root')
    ap.add_argument('--out', required=True, help='output blob path')
    ap.add_argument('--w-act', type=float, default=0.7,
                    help='blend weight for the activations channel (embeddings get the rest)')
    ap.add_argument('--artist-bonus', type=float, default=0.1,
                    help='additive blend boost for albums sharing an artist name')
    ap.add_argument('--atlas-manifest', default=None,
                    help='atlas-manifest.json from the atlas builder (embeds tile positions)')
    args = ap.parse_args()

    tree = Path(args.tree).resolve()
    _, _, classes_json = core.model_paths()
    classes = core.load_classes(classes_json)
    class_index = {c: i for i, c in enumerate(classes)}
    n_cls = len(classes)

    groups = {}  # album dir (posix, without filename) -> list of track dicts
    t0 = time.time()
    n_files = 0
    for p, rel in scan_tracks(tree):
        d = load_track(p)
        if d.get('schema') != schema.SCHEMA_VERSION:
            print(f'WARN: {rel}: schema {d.get("schema")}, skipping', file=sys.stderr)
            continue
        album_dir = rel.rsplit('/', 1)[0] if '/' in rel else ''
        d['_dir'] = album_dir
        groups.setdefault(album_dir, []).append(d)
        n_files += 1
    album_dirs = sorted(groups)
    n = len(album_dirs)
    print(f'{n_files} tracks -> {n} albums (loaded in {time.time()-t0:.0f}s)', flush=True)
    if n < 2:
        raise SystemExit('need at least 2 albums to build a blob')

    act = np.zeros((n, n_cls), dtype=np.float32)
    emb = np.zeros((n, schema.EMB_DIM), dtype=np.float32)
    albums = []
    for i, d in enumerate(album_dirs):
        tracks = groups[d]
        first = tracks[0]
        n_act = np.zeros((len(tracks), n_cls), dtype=np.float32)
        n_emb = np.zeros((len(tracks), schema.EMB_DIM), dtype=np.float32)
        total_dur = 0.0
        for j, t in enumerate(tracks):
            total_dur += float(t.get('duration_sec') or 0)
            acts = t['activations']
            n_act[j] = [acts.get(c, 0.0) for c in classes]
            raw = base64_to_vec(t['embedding']['data'])
            n_emb[j] = raw
        act[i] = np.median(n_act, axis=0)
        emb[i] = np.median(n_emb, axis=0)
        albums.append({'artist': first.get('artist', ''),
                       'album': first.get('album', ''),
                       'track_count': len(tracks),
                       'duration': round(total_dur, 3),
                       'album_dir': d})

    # parent shares: row-normalized activations (sum -> 1; all-zero rows stay 0)
    sums = act.sum(axis=1, keepdims=True)
    parent_shares = np.divide(act, sums, out=np.zeros_like(act), where=sums > 1e-9)
    for a, ps in zip(albums, parent_shares):
        a['parent_shares'] = ps

    sim_act = cosines(act)
    sim_emb = cosines(emb)
    blend = args.w_act * rank_scale(sim_act) + (1.0 - args.w_act) * rank_scale(sim_emb)

    # artist-name bonus (v1 metadata edge)
    artists = np.array([a['artist'].strip().lower() for a in albums])
    same_artist = (artists[:, None] == artists[None, :]) & (artists[:, None] != '')
    blend = blend + args.artist_bonus * same_artist.astype(np.float32)
    np.fill_diagonal(blend, 1.0)

    # top-k by blend, self excluded
    k = min(blobmod.TOP_K, n - 1)
    topk_ids = np.zeros((n, blobmod.TOP_K), dtype=np.uint32)
    topk_sims = np.zeros((n, blobmod.TOP_K), dtype=np.float32)
    for i in range(n):
        row = blend[i].copy()
        row[i] = -np.inf
        idx = np.argpartition(-row, k)[:k]
        idx = idx[np.argsort(-row[idx])]
        topk_ids[i, :k] = idx
        topk_sims[i, :k] = row[idx]

    # stable ids (sha256-based, independent of row order)
    for a in albums:
        a['id_hash'] = blobmod.album_id_hash(a['artist'], a['album_dir'])

    # atlas positions from the atlas builder's manifest (optional)
    images = None
    if args.atlas_manifest:
        mpath = Path(args.atlas_manifest)
        if mpath.exists():
            man = json.load(open(mpath))
            n_sheets = man.get('n_sheets', len(man.get('sheets', [])))
            tiles = [man['albums'].get(d, {'sheet': blobmod.NO_TILE, 'tx': 0, 'ty': 0})
                     for d in album_dirs]
            images = {'n_sheets': n_sheets, 'sheet_dim': man['sheet_dim'],
                      'tile': man['tile'],
                      'tiles': [(t['sheet'], t['tx'], t['ty']) for t in tiles],
                      'sha256': man['sha256']}
            have = sum(1 for t in tiles if t['sheet'] != blobmod.NO_TILE)
            print(f'atlas: {n_sheets} sheet(s), {have}/{n} albums tiled', flush=True)
        else:
            print(f'WARN: atlas manifest not found at {mpath}; blob will have no images',
                  file=sys.stderr, flush=True)

    build_ts = int(time.time())
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.tmp')
    blobmod.write_blob(tmp, build_ts, classes, albums, act, emb,
                       sim_act, sim_emb, blend, (topk_ids, topk_sims), images=images)
    tmp.rename(out)

    mb = out.stat().st_size / 1e6
    print(f'blob: {out} ({mb:.1f} MB), {n} albums, build_ts={build_ts}', flush=True)

    # round-trip verification: decoded vectors must reproduce the stored matrices
    chk = blobmod.read_blob(out)
    assert chk['class_names'] == classes
    assert len(chk['albums']) == n
    assert all(a['id_hash'] for a in chk['albums'])
    if images:
        assert chk['images']['n_sheets'] == images['n_sheets']
        assert sum(1 for s, _, _ in chk['images']['tiles'] if s != blobmod.NO_TILE) == \
            sum(1 for s, _, _ in images['tiles'] if s != blobmod.NO_TILE)
    assert np.allclose(cosines(chk['act_vecs']), sim_act, atol=5e-3)
    assert np.allclose(cosines(chk['emb_vecs']), sim_emb, atol=5e-3)
    top1 = chk['albums'][int(chk['topk_ids'][0][0])]
    print(f'verify: round-trip ok; album[0] "{albums[0]["artist"]} - {albums[0]["album"]}" '
          f'-> top neighbor "{top1["artist"]} - {top1["album"]}" '
          f'({float(chk["topk_sims"][0][0]):.3f})', flush=True)


def base64_to_vec(data: str) -> np.ndarray:
    import base64
    return np.frombuffer(base64.b64decode(data), dtype=np.float32)


if __name__ == '__main__':
    main()
