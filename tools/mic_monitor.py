#!/usr/bin/env python3
"""Which mic is Pupper listening with, and does it hear anything? A page on port 8090.

    python3 tools/mic_monitor.py            # on the Pupper; then open http://<pupper>:8090

It shows PipeWire's default source (what arecord -D default, so robot/gemma_brain.py, uses),
every program recording from a source right now, and the live level of the default source.
It only listens: nothing moves, and no audio leaves the Pupper, only its loudness.
"""
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
RATE, BLOCK = 16000, 1600  # 0.1 s blocks
os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
LEVEL = {"db": -100.0, "peak": 0, "t": 0.0, "error": ""}


def pactl(*args: str) -> str:
    try:
        return subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=3).stdout
    except Exception as e:
        return f"pactl failed: {e}"


def sources() -> dict:
    """The default source, every real source, and who records from which."""
    default = pactl("get-default-source").strip()
    srcs, cur = {}, None
    for line in pactl("list", "sources").splitlines():
        if line.startswith("Source #"):
            cur = {"index": line.split("#")[1]}
        elif cur is not None and ":" in line:
            k, v = (s.strip() for s in line.split(":", 1))
            if k in ("Name", "Description", "State", "Mute") and k.lower() not in cur:
                cur[k.lower()] = v
            if k == "Name":
                srcs[v] = cur
    outs, cur = [], None
    for line in pactl("list", "source-outputs").splitlines():
        if line.startswith("Source Output #"):
            cur = {}
            outs.append(cur)
        elif cur is not None:
            m = re.match(r"\s*(Source|application\.name|application\.process\.binary|application\.process\.id)\s*[:=]\s*\"?([^\"]*)\"?", line)
            if m:
                cur.setdefault(m.group(1), m.group(2))
    by_index = {s["index"]: n for n, s in srcs.items()}
    users = [{"app": o.get("application.name") or o.get("application.process.binary", "?"),
              "binary": o.get("application.process.binary", ""), "pid": o.get("application.process.id", ""),
              "source": by_index.get(o.get("Source", ""), o.get("Source", "?"))}
             for o in outs if "mic_monitor" not in o.get("application.name", "")]
    real = {n: s for n, s in srcs.items() if not n.endswith(".monitor")}
    return {"default": default, "default_info": srcs.get(default, {}), "sources": real, "recording": users}


def meter() -> None:
    """Read the default source forever (parecord, like arecord -D default) and keep its level."""
    while True:
        try:
            p = subprocess.Popen(["parecord", "--raw", "--channels=1", f"--rate={RATE}", "--format=s16le",
                                  "--latency-msec=100", "--client-name=mic_monitor"], stdout=subprocess.PIPE)
            while True:
                data = p.stdout.read(BLOCK * 2)
                if not data:
                    break
                a = array("h", data)
                rms = math.sqrt(sum(x * x for x in a) / len(a))
                LEVEL.update(db=round(20 * math.log10(rms / 32768 + 1e-9), 1),
                             peak=max(abs(x) for x in a), t=time.time(), error="")
            LEVEL["error"] = f"parecord stopped (exit {p.wait()})"
        except Exception as e:
            LEVEL["error"] = str(e)
        time.sleep(1)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pupper Mic</title><style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--mute:#6b6b66;--card:#fff;--line:#e3e3de;--ok:#1a7f4b;--bad:#b42318;--bar:#2f6fde}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecece8;--mute:#9a9a94;--card:#1e1e1c;--line:#2e2e2b;--ok:#4cc38a;--bad:#f97066;--bar:#6c9cff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:720px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--mute);margin:0 0 20px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:14px}
.k{color:var(--mute);font-size:13px}.v{font-size:17px;font-weight:600;word-break:break-all}
.meter{height:22px;background:var(--line);border-radius:6px;overflow:hidden;margin:10px 0 6px}
.fill{height:100%;width:0;background:var(--bar);transition:width .1s}.num{font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:6px 4px;border-top:1px solid var(--line);word-break:break-all}
.ok{color:var(--ok)}.bad{color:var(--bad)}</style></head><body><main>
<h1>Pupper Mic</h1><p class="sub">The mic Pupper listens with, live. Talk or clap: the bar should jump.</p>
<div class="card"><div class="k">Default input (what the Gemma brain's <code>arecord -D default</code> uses)</div>
<div class="v" id="desc">…</div><div class="k" id="name"></div><div id="verdict" style="margin-top:8px"></div></div>
<div class="card"><div class="k">Level of the default input</div><div class="meter"><div class="fill" id="fill"></div></div>
<div class="num"><span id="db">…</span> dBFS · peak <span id="peak">…</span> / 32767 <span id="err" class="bad"></span></div></div>
<div class="card"><div class="k">Programs recording right now</div><table><tbody id="rec"></tbody></table></div>
<div class="card"><div class="k">All inputs on the Pupper</div><table><tbody id="all"></tbody></table></div>
</main><script>
const $=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function level(){try{const l=await (await fetch('level')).json();const pct=Math.max(0,Math.min(100,(l.db+70)/70*100));
$('fill').style.width=pct+'%';$('db').textContent=l.db.toFixed(1);$('peak').textContent=l.peak;
$('err').textContent=l.error?(' · '+l.error):(Date.now()/1000-l.t>2?' · no audio for '+Math.round(Date.now()/1000-l.t)+' s':'');}catch(e){$('err').textContent=' · page lost the Pupper';}}
async function info(){try{const s=await (await fetch('sources')).json();const d=s.default_info||{};
$('desc').textContent=d.description||s.default||'none';$('name').textContent=s.default;
const usb=/usb/i.test(s.default), muted=d.mute==='yes';
$('verdict').innerHTML=muted?'<span class="bad">Muted</span>':usb?'<span class="ok">This is the USB mic on the Pupper.</span>':'<span class="bad">Not the USB mic: check what this device is.</span>';
$('rec').innerHTML=s.recording.length?'<tr><th>Program</th><th>PID</th><th>From</th></tr>'+s.recording.map(r=>`<tr><td>${esc(r.app)} <span class="k">${esc(r.binary)}</span></td><td>${esc(r.pid)}</td><td>${esc(r.source===s.default?'default input':r.source)}</td></tr>`).join(''):'<tr><td class="k">Nothing else is recording (the Gemma brain is not listening).</td></tr>';
$('all').innerHTML=Object.entries(s.sources).map(([n,x])=>`<tr><td>${esc(x.description)}<div class="k">${esc(n)}</div></td><td>${esc(x.state)}</td><td>${x.mute==='yes'?'muted':''}</td></tr>`).join('');}catch(e){}}
setInterval(level,150);setInterval(info,2000);level();info();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/level":
            body, kind = json.dumps(LEVEL), "application/json"
        elif path == "/sources":
            body, kind = json.dumps(sources()), "application/json"
        elif path == "/":
            body, kind = PAGE, "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    threading.Thread(target=meter, daemon=True).start()
    print(f"mic monitor on http://0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
