"""Dot-tracking prototype: a stdlib web app, launchable from VS Code.

    python prototype/dot_tracking/app.py          # then open http://localhost:8000

No Flask, no npm — only the standard library plus numpy, so it runs in the
devcontainer as-is. In a VS Code devcontainer the port is auto-forwarded;
otherwise pass --port.

Layout: the WORLD view on the left is the [-1, 1] grid the target moves in,
with the gaze marker drawn on it. The RETINA inset on the right is the same
instant in gaze-centred coordinates — the dot's offset from the centre of
the field of view. That inset is the one that answers the question: if the
dot stays under the crosshair, the follower is holding it foveated; if it
drifts to the rim, it is not. The strip beneath plots |error| over time so a
transient miss is distinguishable from a standing one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory import SPEC, generate            # noqa: E402
from followers import FOLLOWERS, apply           # noqa: E402


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>dot tracking</title>
<style>
  :root { --fg:#fff; --bg:#000; --dim:#555; --line:#222; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.45
         -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
         -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1120px; margin:0 auto; padding:26px 22px 40px; }
  h1 { font-size:15px; font-weight:600; letter-spacing:.14em;
       text-transform:uppercase; margin:0 0 22px; }
  .controls { display:flex; flex-wrap:wrap; gap:22px; margin-bottom:22px; }
  .group { display:flex; flex-direction:column; gap:7px; }
  .label { font-size:10px; letter-spacing:.16em; text-transform:uppercase;
           color:var(--dim); }
  .seg { display:flex; }
  .seg button { background:var(--bg); color:var(--fg); border:1px solid var(--fg);
                border-right-width:0; padding:6px 13px; font:inherit;
                font-size:12px; cursor:pointer; transition:none; }
  .seg button:last-child { border-right-width:1px; }
  .seg button[aria-pressed="true"] { background:var(--fg); color:var(--bg); }
  .seg button:focus-visible { outline:1px solid var(--fg); outline-offset:2px; }
  .row { display:flex; gap:26px; align-items:flex-start; flex-wrap:wrap; }
  .panel { display:flex; flex-direction:column; gap:8px; }
  canvas { display:block; background:var(--bg); border:1px solid var(--fg); }
  .cap { font-size:10px; letter-spacing:.16em; text-transform:uppercase;
         color:var(--dim); }
  .stats { font-size:12px; color:var(--dim); margin-top:14px;
           font-variant-numeric:tabular-nums; }
  .stats b { color:var(--fg); font-weight:600; }
</style></head><body><div class="wrap">
<h1>dot tracking &mdash; target &amp; gaze</h1>
<div class="controls" id="controls"></div>
<div class="row">
  <div class="panel"><canvas id="world" width="460" height="460"></canvas>
    <div class="cap">world &mdash; grid [-1, 1]</div></div>
  <div class="panel"><canvas id="retina" width="300" height="300"></canvas>
    <div class="cap">retina &mdash; dot relative to gaze centre</div>
    <canvas id="strip" width="300" height="74"></canvas>
    <div class="cap">|error| over time</div></div>
  <div class="panel"><canvas id="joy" width="220" height="220"></canvas>
    <div class="cap">joystick &mdash; top view</div></div>
</div>
<div class="stats" id="stats"></div>
</div><script>
const SPEC = __SPEC__, FOLLOWERS = __FOLLOWERS__;
const sel = {shape:"curve", motion:"continue", speed:"middle", angle:"low",
             follower:"lag"};
let TR = null, k = 0, timer = null;

const C = document.getElementById("controls");
function group(name, opts){
  const g=document.createElement("div"); g.className="group";
  const l=document.createElement("div"); l.className="label"; l.textContent=name;
  const s=document.createElement("div"); s.className="seg";
  opts.forEach(o=>{
    const b=document.createElement("button"); b.textContent=o.replace(/_/g," ");
    b.setAttribute("aria-pressed", sel[name]===o);
    b.onclick=()=>{ sel[name]=o;
      [...s.children].forEach(c=>c.setAttribute("aria-pressed", c===b));
      load(); };
    s.appendChild(b);
  });
  g.append(l,s); C.appendChild(g);
}
Object.entries(SPEC).forEach(([k,v])=>group(k,v));
group("follower", FOLLOWERS);
const g=document.createElement("div"); g.className="group";
g.innerHTML='<div class="label">&nbsp;</div>';
const s=document.createElement("div"); s.className="seg";
const nb=document.createElement("button"); nb.textContent="new seed";
nb.onclick=()=>load(true); s.appendChild(nb); g.appendChild(s); C.appendChild(g);

const W=document.getElementById("world").getContext("2d");
const R=document.getElementById("retina").getContext("2d");
const S=document.getElementById("strip").getContext("2d");
const J=document.getElementById("joy").getContext("2d");
const FOV=0.6;                       // retina half-width, grid units

function toPx(v,size){ return (v+1)/2*(size-2)+1; }

function drawWorld(){
  const n=460; W.clearRect(0,0,n,n); W.fillStyle="#000"; W.fillRect(0,0,n,n);
  W.strokeStyle="#222"; W.lineWidth=1;
  for(let i=1;i<4;i++){ const p=Math.round(i*n/4)+.5;
    W.beginPath(); W.moveTo(p,0); W.lineTo(p,n); W.moveTo(0,p); W.lineTo(n,p); W.stroke(); }
  // travelled path
  W.strokeStyle="#333"; W.beginPath();
  for(let i=0;i<=k;i++){ const px=toPx(TR.x[i],n), py=toPx(-TR.y[i],n);
    i? W.lineTo(px,py) : W.moveTo(px,py); } W.stroke();
  // gaze marker
  const gx=toPx(TR.gx[k],n), gy=toPx(-TR.gy[k],n);
  W.strokeStyle="#888"; W.lineWidth=1;
  W.beginPath(); W.arc(gx,gy,13,0,7); W.stroke();
  W.beginPath(); W.moveTo(gx-19,gy); W.lineTo(gx-15,gy);
  W.moveTo(gx+15,gy); W.lineTo(gx+19,gy); W.moveTo(gx,gy-19); W.lineTo(gx,gy-15);
  W.moveTo(gx,gy+15); W.lineTo(gx,gy+19); W.stroke();
  // the dot
  W.fillStyle="#fff";
  W.beginPath(); W.arc(toPx(TR.x[k],n),toPx(-TR.y[k],n),5,0,7); W.fill();
}

function drawRetina(){
  const n=300, c=n/2; R.fillStyle="#000"; R.fillRect(0,0,n,n);
  R.strokeStyle="#222"; R.lineWidth=1;
  [0.33,0.66,1.0].forEach(f=>{ R.beginPath(); R.arc(c,c,f*(c-2),0,7); R.stroke(); });
  R.strokeStyle="#555";
  R.beginPath(); R.moveTo(c-9,c); R.lineTo(c+9,c);
  R.moveTo(c,c-9); R.lineTo(c,c+9); R.stroke();
  const sx=c+TR.ex[k]/FOV*(c-2), sy=c-TR.ey[k]/FOV*(c-2);
  const inside=Math.hypot(TR.ex[k],TR.ey[k])<=FOV;
  R.fillStyle = inside ? "#fff" : "#666";
  R.beginPath(); R.arc(Math.max(4,Math.min(n-4,sx)),Math.max(4,Math.min(n-4,sy)),
                       inside?6:4,0,7); R.fill();
  if(!inside){ R.fillStyle="#666"; R.font="10px sans-serif";
    R.fillText("outside field of view",8,n-8); }
}

function drawStrip(){
  const w=300,h=74; S.fillStyle="#000"; S.fillRect(0,0,w,h);
  const mx=Math.max(FOV, Math.max(...TR.err));
  S.strokeStyle="#333"; const yf=h-2-(FOV/mx)*(h-6);
  S.setLineDash([3,3]); S.beginPath(); S.moveTo(0,yf); S.lineTo(w,yf); S.stroke();
  S.setLineDash([]);
  S.strokeStyle="#fff"; S.beginPath();
  for(let i=0;i<TR.err.length;i++){
    const px=i/(TR.err.length-1)*w, py=h-2-(TR.err[i]/mx)*(h-6);
    i?S.lineTo(px,py):S.moveTo(px,py); } S.stroke();
  S.strokeStyle="#888"; const cx=k/(TR.err.length-1)*w;
  S.beginPath(); S.moveTo(cx,0); S.lineTo(cx,h); S.stroke();
}

function drawJoy(){
  const n=220, c=n/2, r=c-14;      // r = full-scale deflection, in pixels
  J.fillStyle="#000"; J.fillRect(0,0,n,n);
  // the gate: a white box the stick cannot leave
  J.strokeStyle="#fff"; J.lineWidth=1.5;
  J.strokeRect(c-r,c-r,2*r,2*r);
  J.strokeStyle="#222"; J.lineWidth=1;
  J.beginPath(); J.moveTo(c-r,c); J.lineTo(c+r,c);
  J.moveTo(c,c-r); J.lineTo(c,c+r); J.stroke();
  const jx=TR.jx[k], jy=TR.jy[k];
  const px=c+jx*r, py=c-jy*r;
  // shaft from centre to the stick, then the stick head
  J.strokeStyle="#8b1a1a"; J.lineWidth=2;
  J.beginPath(); J.moveTo(c,c); J.lineTo(px,py); J.stroke();
  J.fillStyle="#e5484d";
  J.beginPath(); J.arc(px,py,7,0,7); J.fill();
  const sat=Math.abs(jx)>=1||Math.abs(jy)>=1;
  if(sat){ J.strokeStyle="#e5484d"; J.lineWidth=1;
           J.strokeRect(c-r-4,c-r-4,2*r+8,2*r+8); }
  J.fillStyle="#555"; J.font="10px sans-serif";
  const sp=Math.hypot(jx,jy)*TR.joy_full_scale;
  J.fillText(sp.toFixed(2)+" u/s"+(sat?"  (saturated)":""), 8, n-8);
}

function stats(){
  const frac=TR.err.filter(e=>e<=FOV).length/TR.err.length*100;
  document.getElementById("stats").innerHTML =
    `mean |error| <b>${TR.err_mean.toFixed(3)}</b> &nbsp; p95 <b>`+
    `${TR.err_p95.toFixed(3)}</b> &nbsp; max <b>${TR.err_max.toFixed(3)}</b>`+
    ` &nbsp;&middot;&nbsp; dot inside the field of view <b>${frac.toFixed(1)}%</b>`+
    ` of the time &nbsp;&middot;&nbsp; stick saturated <b>`+
    `${(TR.joy_sat*100).toFixed(1)}%</b> &nbsp;&middot;&nbsp; seed <b>${TR.settings.seed}</b>`;
}

function frame(){ drawWorld(); drawRetina(); drawStrip(); drawJoy(); stats();
  k=(k+1)%TR.t.length; }

async function load(newseed){
  const q=new URLSearchParams(sel);
  if(!newseed && TR) q.set("seed", TR.settings.seed);
  TR = await (await fetch("/api/trace?"+q)).json();
  k=0; if(timer) clearInterval(timer);
  timer=setInterval(frame, TR.settings.dt*1000);
}
load(true);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):            # keep the console quiet
        pass

    def _send(self, body, ctype):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            page = (PAGE.replace("__SPEC__", json.dumps(SPEC))
                        .replace("__FOLLOWERS__", json.dumps(sorted(FOLLOWERS))))
            return self._send(page, "text/html; charset=utf-8")
        if u.path == "/api/trace":
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                tr = generate(
                    shape=q.get("shape", "curve"),
                    motion=q.get("motion", "continue"),
                    speed=q.get("speed", "middle"),
                    angle=q.get("angle", "low"),
                    duration=float(q.get("duration", 20.0)),
                    seed=int(q["seed"]) if q.get("seed") else None,
                )
                tr.update(apply(q.get("follower", "lag"),
                                tr["t"], tr["x"], tr["y"]))
            except Exception as e:                      # surface, don't hang
                return self._send(json.dumps({"error": f"{type(e).__name__}: {e}"}),
                                  "application/json")
            return self._send(json.dumps(tr), "application/json")
        self.send_error(404)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    a = p.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"dot tracking -> http://localhost:{a.port}   (ctrl-c to stop)")
    print(f"  trajectories: {SPEC}")
    print(f"  followers   : {sorted(FOLLOWERS)}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
