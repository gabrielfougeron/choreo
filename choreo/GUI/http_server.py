import os
import sys
import socketserver
from http.server import SimpleHTTPRequestHandler

def serve_GUI(port = 8000):

    class Handler(SimpleHTTPRequestHandler):
            
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=os.path.dirname(__file__), **kwargs)

        def end_headers(self):

            # Enable Cross-Origin Resource Sharing (CORS)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
            self.send_header('Cross-Origin-Opener-Policy', 'same-origin')

            super().end_headers()

    if sys.version_info < (3, 7, 5):
        # Fix for WASM MIME type for older Python versions
        Handler.extensions_map['.wasm'] = 'application/wasm'

    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving choreo GUI at: http://127.0.0.1:{port}")
        httpd.serve_forever()