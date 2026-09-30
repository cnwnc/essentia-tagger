"""Blob server: serves the similarity blob with ETag/Last-Modified derived from
build_ts so the KAZOOIE proxy can revalidate with conditional GETs (304 = no
re-download). wg-only bind. stdlib http.server; no auth (wg = trust boundary).
TODO: implementation.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--blob', required=True, help='blob file to serve')
    ap.add_argument('--bind', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=9479)
    args = ap.parse_args()
    raise SystemExit('TODO: not implemented yet')


if __name__ == '__main__':
    main()
