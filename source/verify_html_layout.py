"""Check the static paper at phone and desktop viewports with local Edge.

Optional visual check: python source/verify_html_layout.py
Writes screenshots to the system temporary directory.
"""

import base64
import json
import subprocess
import tempfile
import time
from pathlib import Path

import requests
import websocket


EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
URL = (Path(__file__).resolve().parent.parent / "docs" / "index.html").as_uri()


def capture(width: int, height: int, port: int):
    profile = Path(tempfile.gettempdir()) / f"paper-edge-cdp-{port}"
    process = subprocess.Popen([
        str(EDGE), "--headless=new", "--disable-gpu", "--no-first-run",
        f"--user-data-dir={profile}", f"--remote-debugging-port={port}",
        "--remote-allow-origins=*", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        endpoint = f"http://127.0.0.1:{port}/json"
        for _ in range(80):
            try:
                pages = requests.get(endpoint, timeout=.4).json()
                page = next(page for page in pages if page["type"] == "page")
                break
            except (requests.RequestException, StopIteration):
                time.sleep(.15)
        else:
            raise RuntimeError("Edge debugging port did not start")
        ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=20)
        sequence = 0

        def command(method, params=None):
            nonlocal sequence
            sequence += 1
            ws.send(json.dumps({"id": sequence, "method": method, "params": params or {}}))
            while True:
                response = json.loads(ws.recv())
                if response.get("id") == sequence:
                    if "error" in response:
                        raise RuntimeError(response["error"])
                    return response.get("result", {})

        command("Emulation.setDeviceMetricsOverride", {
            "width": width, "height": height, "deviceScaleFactor": 1,
            "mobile": width < 700,
        })
        command("Page.navigate", {"url": URL})
        time.sleep(6)
        metrics = command("Runtime.evaluate", {"expression": "JSON.stringify({viewport: document.documentElement.clientWidth, scrollWidth: document.documentElement.scrollWidth, math: document.querySelectorAll('mjx-container').length, mathErrors: document.querySelectorAll('mjx-merror').length})", "returnByValue": True})
        data = json.loads(metrics["result"]["value"])
        assert width - 20 <= data["viewport"] <= width, data
        assert data["scrollWidth"] == data["viewport"], data
        if width < 700:
            assert data["viewport"] == width, data
        assert data["math"] > 100 and data["mathErrors"] == 0, data
        screenshot = command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        output = Path(tempfile.gettempdir()) / f"paper-{width}.png"
        output.write_bytes(base64.b64decode(screenshot["data"]))
        if width == 390:
            for target, name in (("eq:covode", "equation"), ("fig:ratio", "ratio"), ("fig:work", "work")):
                command("Runtime.evaluate", {"expression": f"document.getElementById('{target}').scrollIntoView({{block: 'start'}})"})
                time.sleep(1.2)
                shot = command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
                (Path(tempfile.gettempdir()) / f"paper-390-{name}.png").write_bytes(base64.b64decode(shot["data"]))
            assets = command("Runtime.evaluate", {"expression": "JSON.stringify([...document.images].map(i => ({loaded: i.complete && i.naturalWidth > 0, src: i.currentSrc})))", "returnByValue": True})
            images = json.loads(assets["result"]["value"])
            assert len(images) == 2 and all(image["loaded"] and image["src"].endswith("mobile.svg") for image in images), images
        ws.close()
        print(f"{width}px: no page overflow, {data['math']} MathJax nodes, no math errors; {output}")
    finally:
        process.terminate()


if __name__ == "__main__":
    capture(320, 760, 9236)
    capture(390, 844, 9237)
    capture(1440, 900, 9238)
