"""Blob server: GET/HEAD for the similarity blob with ETag/Last-Modified so
the KAZOOIE proxy can revalidate with conditional GETs (304 = no re-download).
stdlib http.server; no auth — bind to the wireguard interface (wg = trust
boundary). Single file, so ETag = size-mtime and reads are mmap-free.
"""
import argparse
import email.utils
import http.server
import os
import threading


class BlobState:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.refresh()

    def refresh(self):
        st = os.stat(self.path)
        with self.lock:
            self.size = st.st_size
            self.mtime = int(st.st_mtime)
            self.etag = f'"{self.size:x}-{self.mtime:x}"'
            self.last_modified = email.utils.formatdate(self.mtime, usegmt=True)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    blob: BlobState = None  # set by serve()

    def _send_headers(self, code, length=None):
        b = self.blob
        self.send_response(code)
        self.send_header('ETag', b.etag)
        self.send_header('Last-Modified', b.last_modified)
        self.send_header('Cache-Control', 'public, max-age=300')
        self.send_header('Accept-Ranges', 'none')
        if length is not None:
            self.send_header('Content-Length', str(length))
        self.end_headers()

    def _handle(self, head_only: bool):
        b = self.blob
        try:
            b.refresh()
        except OSError:
            self._send_headers(404, 0)
            return
        inm = self.headers.get('If-None-Match')
        ims = self.headers.get('If-Modified-Since')
        not_modified = (inm and inm.strip() == b.etag) or \
                       (inm is None and ims and b.last_modified == ims)
        if not_modified:
            self._send_headers(304, 0)
            return
        self._send_headers(200, b.size)
        if head_only:
            return
        with open(b.path, 'rb') as f:
            while chunk := f.read(1 << 20):
                self.wfile.write(chunk)

    def do_GET(self):
        if self.path in ('/', '/blob', '/albums'):
            self._handle(head_only=False)
        else:
            self._send_headers(404, 0)

    def do_HEAD(self):
        if self.path in ('/', '/blob', '/albums'):
            self._handle(head_only=True)
        else:
            self._send_headers(404, 0)

    def log_message(self, fmt, *args):
        print(f'{self.address_string()} {fmt % args}', flush=True)


def serve(blob_path, bind, port):
    Handler.blob = BlobState(blob_path)
    http.server.ThreadingHTTPServer((bind, port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--blob', required=True, help='blob file to serve')
    ap.add_argument('--bind', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=9478)
    args = ap.parse_args()
    print(f'serving {args.blob} on {args.bind}:{args.port}', flush=True)
    serve(args.blob, args.bind, args.port)


if __name__ == '__main__':
    main()
