"""Blob builder: classifier JSON tree -> similarity blob (Contract D).

Pure numpy, host-agnostic, cheap to re-run (weights/aggregation are rebuild
knobs; re-extraction is never needed). Steps: group tracks by parent dir
(album) -> element-wise median per album (activations + embedding) ->
parent-share axes (sum within album, share cross-album, never mean) ->
per-channel cosine matrices -> embeddings percentile-scaled before blending
(activations-dominant default; tuned on full data) -> artist-name-match bonus
-> dense matrices + top-k -> blob.
TODO: implementation.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--tree', required=True, help='classifier JSON tree root')
    ap.add_argument('--out', required=True, help='output blob path')
    args = ap.parse_args()
    raise SystemExit('TODO: not implemented yet')


if __name__ == '__main__':
    main()
