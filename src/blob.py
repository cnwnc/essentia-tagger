"""Versioned binary blob format (Contract D in ARCHITECTURE.md).

Container: versioned little-endian, gzip-compressed on the wire, all offsets
u64, strings UTF-8 with u32 length. Layout (see ARCHITECTURE.md for detail):

  magic "ESSG" u32 | format_version u32 | flags u32 | build_ts u64
  counts: n_albums u32, n_classes u32, n_emb_dim u32
  class_names[n_classes]
  album_table[n_albums]: id u32, artist str, album str, track_count u32,
                         duration f32, parent_shares[n_classes] f16
  vectors: act_vecs[n × 519] f16, emb_vecs[n × 768] f16
  matrices: sim_act[n²] f16, sim_emb[n²] f16, sim_blend[n²] f16
  topk: per album, k=32 (id u32, sim f16)
  footer: u32 crc32 of payload | magic "ESSG"

Consumed client-side by the frontend (one fetch per session, everything else
computed locally: thresholds, reweighting, radio walk).
TODO: reader/writer implementation.
"""

MAGIC = b'ESSG'
FORMAT_VERSION = 1
TOP_K = 32
