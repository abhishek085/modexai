"""`modexai ui`: the exchange web page, served from localhost with your own drafts merged in."""

from __future__ import annotations

import contextlib
import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from . import config, hardware, notes, recipes
from .registry import load_registry


def _registry_doc() -> bytes:
    reg = load_registry()
    return json.dumps({**reg.meta, "models": [m.to_dict() for m in reg.models]}).encode()


def _recipes_doc() -> bytes:
    shared = recipes.load_index()
    ids = {r["id"] for r in shared}
    mine = [{**r, "_mine": True} for r in recipes.load_local() if r["id"] not in ids]
    return json.dumps({"recipes": mine + shared, "this_machine": hardware.label()}).encode()


def _notes_doc() -> bytes:
    return json.dumps({"notes": notes.load_index()}).encode()


DOCS = {"/registry.json": _registry_doc, "/recipes.json": _recipes_doc, "/notes.json": _notes_doc}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(config.WEB_DIR), **kw)

    def log_message(self, *_):  # quiet
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in DOCS:
            body = DOCS[path]()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def serve(port: int = 8484, open_browser: bool = True) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"ModexAI exchange on {url}  (localhost only · Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    with contextlib.suppress(KeyboardInterrupt):
        httpd.serve_forever()
