"""Sync client: diff library vs classifier tree, push new albums to the
classifier server, store results. Idempotent oneshot (timer-friendly).

Flow: diff (mtime+size, album as unit) -> transcode-stream tarball to
POST /classify (GPU warms during transfer) -> unpack response into the
classifier tree -> log in-band errors locally.
TODO: implementation.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--music', required=True, help='music library root')
    ap.add_argument('--tree', required=True, help='classifier JSON tree root')
    ap.add_argument('--url', required=True, help='classifier server base URL')
    args = ap.parse_args()
    raise SystemExit('TODO: not implemented yet')


if __name__ == '__main__':
    main()
