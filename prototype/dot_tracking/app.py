"""Dot-tracking prototype: a stdlib web app, launchable from VS Code.

    python prototype/dot_tracking/app.py          # then open http://localhost:8000

No Flask, no npm — only the standard library plus numpy, so it runs in the
devcontainer as-is. In a VS Code devcontainer the port is auto-forwarded;
otherwise pass --port.

Layout: the WORLD view on the left is the [-1, 1] grid the target moves in,
with the gaze marker drawn on it. The RETINA inset is the same instant in
gaze-centred coordinates — the dot's offset from the centre of the field of
view. That inset is the one that answers the question: if the dot stays under
the crosshair, the controller is holding it foveated; if it drifts to the rim,
it is not. The strip beneath plots |error| over time so a transient miss is
distinguishable from a standing one. The JOYSTICK panel shows the stick that
produced the gaze, top view.

The controller selector and its sliders are built from
``followers.PARAMS``, so adding a controller with its knobs needs no edit
here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory import SPEC, generate                   # noqa: E402
from followers import FOLLOWERS, PARAMS, apply          # noqa: E402


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>dot tracking</title>
<style>
  :root { --fg:#fff; --bg:#000; --dim:#fff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.45
         -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
         -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1180px; margin:0 auto; padding:26px 22px 40px; }
  h1 { font-size:15px; font-weight:600; letter-spacing:.14em;
       text-transform:uppercase; margin:0 0 22px; }
  .controls { display:flex; flex-wrap:wrap; gap:22px; margin-bottom:18px; }
  .group { display:flex; flex-direction:column; gap:7px; }
  .label { font-size:10px; letter-spacing:.16em; text-transform:uppercase;
           color:var(--dim); }
  .seg { display:flex; }
  .seg button { background:var(--bg); color:var(--fg); border:1px solid var(--fg);
                border-right-width:0; padding:6px 13px; font:inherit;
                font-size:12px; cursor:pointer; }
  .seg button:last-child { border-right-width:1px; }
  .seg button[aria-pressed="true"] { background:var(--fg); color:var(--bg); }
  .knobs { display:flex; flex-wrap:wrap; gap:26px; margin:0 0 22px;
           padding:14px 16px; border:1px solid #333; }
  .knob { display:flex; flex-direction:column; gap:5px; min-width:210px; }
  .knob .kl { font-size:11px; color:var(--dim); display:flex;
              justify-content:space-between; gap:12px; }
  .knob .kl b { color:var(--fg); font-weight:600;
                font-variant-numeric:tabular-nums; }
  .knob .ends { display:flex; justify-content:space-between; font-size:9px;
                color:var(--fg); font-variant-numeric:tabular-nums; }
  input[type=range] { -webkit-appearance:none; appearance:none; width:100%;
                      height:1px; background:var(--fg); outline:none; margin:6px 0; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance:none;
    appearance:none; width:13px; height:13px; background:var(--fg);
    border:1px solid var(--fg); cursor:pointer; border-radius:0; }
  input[type=range]::-moz-range-thumb { width:13px; height:13px;
    background:var(--fg); border:1px solid var(--fg); cursor:pointer;
    border-radius:0; }
  .row { display:flex; gap:26px; align-items:flex-start; flex-wrap:wrap; }
  .panel { display:flex; flex-direction:column; gap:8px; }
  canvas { display:block; background:var(--bg); border:1px solid var(--fg); }
  .cap { font-size:10px; letter-spacing:.16em; text-transform:uppercase;
         color:var(--dim); }
  .stats { font-size:12px; color:var(--dim); margin-top:16px;
           font-variant-numeric:tabular-nums; }
  .stats b { color:var(--fg); font-weight:600; }
</style></head><body><div class="wrap">
<h1>dot tracking &mdash; target &amp; gaze</h1>
<div class="controls" id="controls"></div>
<div class="knobs" id="knobs"></div>
<div class="row">
  <div class="panel"><canvas id="world" width="460" height="460"></canvas>
    <div class="cap">world &mdash; grid [-1, 1]</div></div>
  <div class="panel"><canvas id="retina" width="300" height="300"></canvas>
    <div class="cap">retina &mdash; dot relative to gaze centre</div>
    <canvas id="strip" width="300" height="74"></canvas>
    <div class="cap">|error| over time</div></div>
  <div class="panel"><canvas id="joy" width="260" height="260"></canvas>
    <div class="cap">joystick &mdash; top view (&times;__JOYGAIN__ view gain)</div></div>
</div>
<div class="stats" id="stats"></div>
</div><script>
const SPEC=__SPEC__, FOLLOWERS=__FOLLOWERS__, PARAMS=__PARAMS__;
const JOYGAIN=__JOYGAIN__;
const DURATIONS=["4","8","16","30"];
const sel={start:"random",shape:"curve",motion:"continue",speed:"middle",
           angle:"low",follower:"pursuit",duration:"16"};
const knob={};                       // current parameter values
let TR=null, k=0, timer=null, pending=null;

const C=document.getElementById("controls"), K=document.getElementById("knobs");
function group(name,opts,onpick){
  const g=document.createElement("div"); g.className="group";
  const l=document.createElement("div"); l.className="label"; l.textContent=name;
  const s=document.createElement("div"); s.className="seg";
  opts.forEach(o=>{
    const b=document.createElement("button");
    b.textContent=(name==="duration"?o+" s":o.replace(/_/g," "));
    b.setAttribute("aria-pressed", sel[name]===o);
    b.onclick=()=>{ sel[name]=o;
      [...s.children].forEach(c=>c.setAttribute("aria-pressed",c===b));
      if(onpick) onpick(o); load(); };
    s.appendChild(b);
  });
  g.append(l,s); C.appendChild(g); return s;
}
Object.entries(SPEC).forEach(([n,v])=>group(n,v));
group("duration",DURATIONS);
group("follower",FOLLOWERS,buildKnobs);
const gx=document.createElement("div"); gx.className="group";
gx.innerHTML='<div class="label">&nbsp;</div>';
const sx=document.createElement("div"); sx.className="seg";
const nb=document.createElement("button"); nb.textContent="new seed";
nb.onclick=()=>load(true); sx.appendChild(nb); gx.appendChild(sx); C.appendChild(gx);

function buildKnobs(name){
  K.innerHTML=""; const ps=PARAMS[name]||[];
  if(!ps.length){ K.innerHTML='<div class="label">no parameters</div>'; return; }
  ps.forEach(p=>{
    if(knob[p.name]===undefined) knob[p.name]=p.default;
    const d=document.createElement("div"); d.className="knob";
    const lab=document.createElement("div"); lab.className="kl";
    const val=document.createElement("b"); val.textContent=(+knob[p.name]).toFixed(2);
    const nm=document.createElement("span"); nm.textContent=p.label;
    lab.append(nm,val);
    const r=document.createElement("input"); r.type="range";
    r.min=p.min; r.max=p.max; r.step=p.step; r.value=knob[p.name];
    const ends=document.createElement("div"); ends.className="ends";
    ends.innerHTML=`<span>${p.min}</span><span>${p.max}</span>`;
    r.oninput=()=>{ knob[p.name]=+r.value; val.textContent=(+r.value).toFixed(2);
                    clearTimeout(pending); pending=setTimeout(()=>load(),110); };
    d.append(lab,r,ends); K.appendChild(d);
  });
}
buildKnobs(sel.follower);

const W=document.getElementById("world").getContext("2d");
const R=document.getElementById("retina").getContext("2d");
const S=document.getElementById("strip").getContext("2d");
const J=document.getElementById("joy").getContext("2d");
const FOV=0.6;                       // retina half-width, grid units
const TRAIL=110;                     // samples drawn as the bright recent track

function toPx(v,size){ return (v+1)/2*(size-2)+1; }

function drawWorld(){
  const n=460; W.fillStyle="#000"; W.fillRect(0,0,n,n);
  W.strokeStyle="#1c1c1c"; W.lineWidth=1;
  for(let i=1;i<4;i++){ const p=Math.round(i*n/4)+.5;
    W.beginPath(); W.moveTo(p,0); W.lineTo(p,n); W.moveTo(0,p); W.lineTo(n,p); W.stroke(); }
  // the whole track so far, then the recent stretch brighter and thicker,
  // so both the shape of the path and the current direction of travel read.
  W.lineJoin="round"; W.lineCap="round";
  W.strokeStyle="#5a5a5a"; W.lineWidth=2; W.beginPath();
  for(let i=0;i<=k;i++){ const px=toPx(TR.x[i],n), py=toPx(-TR.y[i],n);
    i?W.lineTo(px,py):W.moveTo(px,py); } W.stroke();
  W.strokeStyle="#d8d8d8"; W.lineWidth=3.5; W.beginPath();
  for(let i=Math.max(0,k-TRAIL);i<=k;i++){ const px=toPx(TR.x[i],n), py=toPx(-TR.y[i],n);
    i===Math.max(0,k-TRAIL)?W.moveTo(px,py):W.lineTo(px,py); } W.stroke();
  // gaze marker
  const cx=toPx(TR.gx[k],n), cy=toPx(-TR.gy[k],n);
  W.strokeStyle="#888"; W.lineWidth=1.2;
  W.beginPath(); W.arc(cx,cy,14,0,7); W.stroke();
  W.beginPath(); W.moveTo(cx-21,cy); W.lineTo(cx-16,cy); W.moveTo(cx+16,cy);
  W.lineTo(cx+21,cy); W.moveTo(cx,cy-21); W.lineTo(cx,cy-16);
  W.moveTo(cx,cy+16); W.lineTo(cx,cy+21); W.stroke();
  // the target
  W.fillStyle="#fff";
  W.beginPath(); W.arc(toPx(TR.x[k],n),toPx(-TR.y[k],n),7,0,7); W.fill();
}

function drawRetina(){
  const n=300,c=n/2; R.fillStyle="#000"; R.fillRect(0,0,n,n);
  R.strokeStyle="#242424"; R.lineWidth=1;
  [0.33,0.66,1.0].forEach(f=>{ R.beginPath(); R.arc(c,c,f*(c-2),0,7); R.stroke(); });
  R.strokeStyle="#777"; R.lineWidth=1.2;
  R.beginPath(); R.moveTo(c-10,c); R.lineTo(c+10,c);
  R.moveTo(c,c-10); R.lineTo(c,c+10); R.stroke();
  const px=c+TR.ex[k]/FOV*(c-2), py=c-TR.ey[k]/FOV*(c-2);
  const inside=Math.hypot(TR.ex[k],TR.ey[k])<=FOV;
  R.fillStyle=inside?"#fff":"#777";
  R.beginPath(); R.arc(Math.max(5,Math.min(n-5,px)),Math.max(5,Math.min(n-5,py)),
                       inside?8:5,0,7); R.fill();
  if(!inside){ R.fillStyle="#fff"; R.font="10px sans-serif";
    R.fillText("outside field of view",8,n-8); }
}

function drawStrip(){
  const w=300,h=74; S.fillStyle="#000"; S.fillRect(0,0,w,h);
  const mx=Math.max(FOV,Math.max(...TR.err));
  S.strokeStyle="#444"; const yf=h-2-(FOV/mx)*(h-6);
  S.setLineDash([3,3]); S.beginPath(); S.moveTo(0,yf); S.lineTo(w,yf); S.stroke();
  S.setLineDash([]);
  S.strokeStyle="#fff"; S.lineWidth=1.5; S.beginPath();
  for(let i=0;i<TR.err.length;i++){
    const px=i/(TR.err.length-1)*w, py=h-2-(TR.err[i]/mx)*(h-6);
    i?S.lineTo(px,py):S.moveTo(px,py); } S.stroke();
  S.strokeStyle="#888"; S.lineWidth=1; const cx=k/(TR.err.length-1)*w;
  S.beginPath(); S.moveTo(cx,0); S.lineTo(cx,h); S.stroke();
}

function drawJoy(){
  const n=260,c=n/2,r=c-16;
  J.fillStyle="#000"; J.fillRect(0,0,n,n);
  // the gate: a filled white box the stick cannot leave
  J.fillStyle="#fff"; J.fillRect(c-r,c-r,2*r,2*r);
  J.strokeStyle="#c9c9c9"; J.lineWidth=1;
  J.beginPath(); J.moveTo(c-r,c); J.lineTo(c+r,c);
  J.moveTo(c,c-r); J.lineTo(c,c+r); J.stroke();
  // View gain only — the underlying command is unchanged. Small deflections
  // are otherwise invisible because a well-tuned controller rarely asks for
  // more than a fraction of full scale.
  const jx=Math.max(-1,Math.min(1,TR.jx[k]*JOYGAIN));
  const jy=Math.max(-1,Math.min(1,TR.jy[k]*JOYGAIN));
  const px=c+jx*r, py=c-jy*r;
  J.strokeStyle="#e5484d"; J.lineWidth=6; J.lineCap="round";
  J.beginPath(); J.moveTo(c,c); J.lineTo(px,py); J.stroke();
  J.fillStyle="#e5484d";
  J.beginPath(); J.arc(px,py,13,0,7); J.fill();
  J.fillStyle="#222";                       // black text, on the white gate
  J.beginPath(); J.arc(c,c,3.5,0,7); J.fill();
  const sat=Math.abs(TR.jx[k])>=1||Math.abs(TR.jy[k])>=1;
  const sp=Math.hypot(TR.jx[k],TR.jy[k])*TR.joy_full_scale;
  J.fillStyle="#111"; J.font="600 11px sans-serif";
  J.fillText(sp.toFixed(2)+" u/s"+(sat?"   SATURATED":""), c-r+8, c+r-9);
  if(sat){ J.strokeStyle="#e5484d"; J.lineWidth=2;
           J.strokeRect(c-r+1,c-r+1,2*r-2,2*r-2); }
}

function stats(){
  const frac=TR.err.filter(e=>e<=FOV).length/TR.err.length*100;
  document.getElementById("stats").innerHTML=
    `mean |error| <b>${TR.err_mean.toFixed(3)}</b> &nbsp; p95 <b>`+
    `${TR.err_p95.toFixed(3)}</b> &nbsp; max <b>${TR.err_max.toFixed(3)}</b>`+
    ` &nbsp;&middot;&nbsp; dot inside the field of view <b>${frac.toFixed(1)}%</b>`+
    ` of the time &nbsp;&middot;&nbsp; stick saturated <b>`+
    `${(TR.joy_sat*100).toFixed(1)}%</b> &nbsp;&middot;&nbsp; seed <b>`+
    `${TR.settings.seed}</b>`;
}

function frame(){ drawWorld(); drawRetina(); drawStrip(); drawJoy(); stats();
  k=(k+1)%TR.t.length; }

async function load(newseed){
  const q=new URLSearchParams(sel);
  Object.entries(knob).forEach(([n,v])=>q.set(n,v));
  if(!newseed && TR) q.set("seed", TR.settings.seed);
  const keep = (!newseed && TR) ? k : 0;
  const r=await (await fetch("/api/trace?"+q)).json();
  if(r.error){ document.getElementById("stats").textContent=r.error; return; }
  TR=r; k=Math.min(keep, TR.t.length-1);
  if(timer) clearInterval(timer);
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
                        .replace("__FOLLOWERS__", json.dumps(sorted(FOLLOWERS)))
                        .replace("__PARAMS__", json.dumps(PARAMS))
                        .replace("__JOYGAIN__", str(JOY_VIEW_GAIN)))
            return self._send(page, "text/html; charset=utf-8")
        if u.path == "/api/trace":
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                tr = generate(
                    start=q.get("start", "random"),
                    shape=q.get("shape", "curve"),
                    motion=q.get("motion", "continue"),
                    speed=q.get("speed", "middle"),
                    angle=q.get("angle", "low"),
                    duration=float(q.get("duration", 20.0)),
                    seed=int(q["seed"]) if q.get("seed") else None,
                )
                name = q.get("follower", "pursuit")
                knobs = {p["name"]: float(q[p["name"]])
                         for p in PARAMS.get(name, []) if p["name"] in q}
                tr.update(apply(name, tr["t"], tr["x"], tr["y"], **knobs))
            except Exception as e:                      # surface, don't hang
                return self._send(json.dumps({"error": f"{type(e).__name__}: {e}"}),
                                  "application/json")
            return self._send(json.dumps(tr), "application/json")
        self.send_error(404)


# Display-only magnification of the stick deflection. A well-tuned controller
# spends most of its time well inside the gate, so the raw trace barely leaves
# the centre; this scales the drawn position (and only that) so the course is
# legible. Saturation is still reported from the UNSCALED command.
JOY_VIEW_GAIN = 3.0


def main():
    global JOY_VIEW_GAIN
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--joy-view-gain", type=float, default=JOY_VIEW_GAIN,
                   help="display magnification of the stick deflection")
    a = p.parse_args()
    JOY_VIEW_GAIN = a.joy_view_gain
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"dot tracking -> http://localhost:{a.port}   (ctrl-c to stop)")
    print(f"  trajectories: {SPEC}")
    print(f"  controllers : {sorted(FOLLOWERS)}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
