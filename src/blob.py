"""Versioned binary similarity blob (Contract D in ARCHITECTURE.md).

Container: little-endian, gzip-compressed as a whole. Layout:

  header:  magic "ESSG" | format_version u32 | flags u32 | build_ts u64
           n_albums u32 | n_classes u32 | n_emb_dim u32
  strings: class_names[n_classes]                      (u32 len + utf8)
  albums:  per album: id u32 (= index), artist str, album str,
                      track_count u32, duration f32,
                      parent_shares f16[n_classes]     (row-normalized activations)
  vectors: act f16[n x n_classes] | emb f16[n x n_emb_dim]
  matrices: sim_act f16[n*n] | sim_emb f16[n*n] | sim_blend f16[n*n]
           (raw cosines for act/emb; blend = rank-scaled 0.7/0.3 + artist bonus)
  topk:    per album k=32 pairs (id u32, sim f16)      (by blend, self excluded)
  footer:  crc32 u32 (of everything before it) | magic "ESSG"

All numeric fields native little-endian; f16 via numpy (x86 = LE).
The whole file is one gzip stream; the browser decompresses natively.
"""
import gzip
import struct
import zlib

import numpy as np

MAGIC = b'ESSG'
FORMAT_VERSION = 1
TOP_K = 32
HEADER = struct.Struct('<IIQIII')  # version, flags, build_ts, n, n_cls, n_dim (magic is raw bytes)


def _wstr(out: list, s: str):
    b = s.encode('utf-8')
    out.append(struct.pack('<I', len(b)))
    out.append(b)


def _warr(out: list, a: np.ndarray):
    a = np.ascontiguousarray(a)
    if a.dtype.kind == 'f':
        a = a.astype('<f2')
    out.append(a.tobytes())


def write_blob(path, build_ts, class_names, albums, act_vecs, emb_vecs,
               sim_act, sim_emb, sim_blend, topk) -> None:
    """albums: list of dicts {artist, album, track_count, duration, parent_shares}
    topk: tuple (ids u32 [n, TOP_K], sims f32 [n, TOP_K])."""
    n = len(albums)
    n_cls = len(class_names)
    n_dim = int(emb_vecs.shape[1])
    topk_ids, topk_sims = topk
    body = []

    body.append(MAGIC + HEADER.pack(FORMAT_VERSION, 0, int(build_ts), n, n_cls, n_dim))
    for c in class_names:
        _wstr(body, c)
    for i, a in enumerate(albums):
        body.append(struct.pack('<I', i))
        _wstr(body, a['artist'])
        _wstr(body, a['album'])
        body.append(struct.pack('<I', int(a['track_count'])))
        body.append(struct.pack('<f', float(a['duration'])))
        _warr(body, a['parent_shares'])
    _warr(body, act_vecs)
    _warr(body, emb_vecs)
    _warr(body, sim_act)
    _warr(body, sim_emb)
    _warr(body, sim_blend)
    _warr(body, topk_ids.astype('<u4').reshape(-1))
    _warr(body, topk_sims.astype('<f2').reshape(-1))

    payload = b''.join(body)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    blob = payload + struct.pack('<I', crc) + MAGIC
    with open(path, 'wb') as f:
        f.write(gzip.compress(blob, compresslevel=6, mtime=0))


def read_blob(path, to_f32: bool = True) -> dict:
    """Parse a blob file. Returns dict with all fields as numpy arrays/lists."""
    with open(path, 'rb') as f:
        raw = gzip.decompress(f.read())
    if raw[:4] != MAGIC or raw[-4:] != MAGIC:
        raise ValueError('bad magic')
    crc_stored = struct.unpack('<I', raw[-8:-4])[0]
    if zlib.crc32(raw[:-8]) & 0xFFFFFFFF != crc_stored:
        raise ValueError('crc mismatch')
    buf = raw[:-8]
    if raw[:4] != MAGIC:
        raise ValueError('bad header magic')
    off = 4

    def take(fmt):
        nonlocal off
        v = struct.unpack_from(fmt, buf, off)
        off += struct.calcsize(fmt)
        return v

    def rstr():
        nonlocal off
        (ln,) = take('<I')
        s = buf[off:off + ln].decode('utf-8')
        off += ln
        return s

    def rarr(shape, dtype):
        nonlocal off
        count = int(np.prod(shape)) if shape else 1
        a = np.frombuffer(buf, dtype='<f2' if dtype == np.float16 else dtype,
                          count=count, offset=off)
        off += a.nbytes
        a = a.reshape(shape)
        return a.astype(np.float32) if to_f32 else a

    version, flags, build_ts, n, n_cls, n_dim = HEADER.unpack_from(buf, off)
    off += HEADER.size
    if version != FORMAT_VERSION:
        raise ValueError(f'unsupported blob version {version}')
    class_names = [rstr() for _ in range(n_cls)]
    albums = []
    for i in range(n):
        (idx,) = take('<I')
        assert idx == i, f'album index {idx} != {i}'
        artist, album = rstr(), rstr()
        (track_count,) = take('<I')
        (duration,) = take('<f')
        parent_shares = rarr((n_cls,), np.float16)
        albums.append({'artist': artist, 'album': album, 'track_count': track_count,
                       'duration': duration, 'parent_shares': parent_shares})
    act_vecs = rarr((n, n_cls), np.float16)
    emb_vecs = rarr((n, n_dim), np.float16)
    sim_act = rarr((n, n), np.float16)
    sim_emb = rarr((n, n), np.float16)
    sim_blend = rarr((n, n), np.float16)
    topk_ids = rarr((n, TOP_K), np.uint32)
    topk_sims = rarr((n, TOP_K), np.float16)
    if off != len(buf):
        raise ValueError(f'trailing bytes: off={off} len={len(buf)}')
    return {'version': version, 'flags': flags, 'build_ts': build_ts,
            'class_names': class_names, 'albums': albums,
            'act_vecs': act_vecs, 'emb_vecs': emb_vecs,
            'sim_act': sim_act, 'sim_emb': sim_emb, 'sim_blend': sim_blend,
            'topk_ids': topk_ids, 'topk_sims': topk_sims}
