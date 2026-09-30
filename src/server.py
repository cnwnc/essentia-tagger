"""Resident classifier server: POST /classify (zstd tar of audio in -> tar of
track JSONs out). The ONLY functional route; GET /health is liveness-only.

Auto-warm: a cold request loads models in-line (~15-30s) then processes; after
~10min idle the models unload (idle cost ~0) and the next request re-warms.
Stateless: request tarball is staged in /dev/shm and deleted after response.
Per-track errors are in-band (.errors.jsonl in the response tar); 4xx/5xx only
for whole-batch failure. No auth: bind to the wireguard IP (wg = trust boundary).
TODO: implementation (stdlib http.server + core).
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--bind', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=9478)
    args = ap.parse_args()
    raise SystemExit('TODO: not implemented yet')


if __name__ == '__main__':
    main()
