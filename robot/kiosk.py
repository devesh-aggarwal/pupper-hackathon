"""Drive the kiosk browser on the robot's screen through Chromium's local debugging port.

The robot's screen has no keyboard or touch input, so this writes the Gemini key
and on-robot settings into the page's storage before the app loads, presses
WAKE PUPPER, and presses it again whenever the Gemini session drops.
Usage (start.sh runs it):  python3 robot/kiosk.py <api-key-file>
"""
import asyncio, json, sys, time, urllib.request

import websockets

SETTINGS = {"robotApiUrl": "", "cameraSource": "server", "eyeTrackingSource": "server",
            "enableVideo": True, "enableAudio": True}


def page_ws():
    for _ in range(60):
        try:
            for t in json.load(urllib.request.urlopen("http://127.0.0.1:9222/json")):
                if t.get("type") == "page" and "localhost:8000" in t.get("url", ""):
                    return t["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(1)
    sys.exit("kiosk: the app page never appeared in Chromium")


class Page:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method, params=None):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params or {}}))
        while True:
            m = json.loads(await self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})

    async def ev(self, expr):
        r = await self.call("Runtime.evaluate", {"expression": expr, "returnByValue": True})
        return r.get("result", {}).get("value")


async def main(key_file):
    key = open(key_file).read().strip()
    setup = f"""(() => {{ try {{
      const cfg = Object.assign(JSON.parse(localStorage.getItem('pupster_config') || '{{}}'), {json.dumps(SETTINGS)});
      localStorage.setItem('pupster_api_key', {json.dumps(key)});
      localStorage.setItem('pupster_config', JSON.stringify(cfg));
    }} catch (e) {{}} }})();"""
    async with websockets.connect(page_ws(), max_size=None) as ws:
        p = Page(ws)
        await p.call("Page.enable")
        await p.call("Page.addScriptToEvaluateOnNewDocument", {"source": setup})
        await p.call("Page.reload", {"ignoreCache": True})
        await asyncio.sleep(5)
        ok = await p.ev("(localStorage.getItem('pupster_api_key') || '').length > 0")
        print("kiosk: settings applied" if ok else "kiosk: WARNING settings not applied", flush=True)
    while True:  # press Wake whenever it shows, e.g. after the Gemini session drops
        try:
            async with websockets.connect(page_ws(), max_size=None) as ws:
                p = Page(ws)
                while True:
                    if await p.ev("""(() => { const b = [...document.querySelectorAll('button')]
                            .find(b => b.textContent.includes('WAKE PUPPER'));
                            if (b) { b.click(); return true; } return false; })()"""):
                        print(time.strftime("%H:%M:%S"), "kiosk: pressed WAKE PUPPER", flush=True)
                        await asyncio.sleep(10)
                    await asyncio.sleep(3)
        except Exception as e:
            print("kiosk: reconnecting after", type(e).__name__, flush=True)
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
