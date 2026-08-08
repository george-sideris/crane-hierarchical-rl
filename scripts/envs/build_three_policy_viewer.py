#!/usr/bin/env python3
"""Interactive 3D comparison of the THREE policy families on every real decision cloud.

  BASELINE (deployed heuristic)   - argmax highest supported point, tight crop @1024
  REGRESSION BC (bc_aug1v2_dig25) - MSE coordinate head,           tight crop @1024
  SCORING BC (P2c)                - per-point argmax head,         margin crop @2048

Each slide marks which policy was ACTUALLY DEPLOYED for that cycle (thick white ring + label)
and shows what the other two WOULD have commanded on the same cloud.

CROPS (this is the honest part): the three families were trained on different observation
crops, so a fair replay feeds each its OWN crop derived from the same raw bag cloud.
Slides from runs whose recording only kept the tight crop cannot supply the scoring policy's
margin band; those are marked TIGHT-ONLY and the scoring marker there is off-distribution.

    python3 scripts/envs/build_three_policy_viewer.py --out docs/figures/three_policy_3d.html
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "fpi_crane_ros2", "fpi_crane_rl", "fpi_crane_rl"))
from scoring_head import ScoringGraspPolicy                      # noqa: E402
from render_policy_comparison import RegressionBC, dec5, fps, crop, pad_to, B_MIN, B_MAX  # noqa: E402
from policy_loader import HeuristicPolicy                        # noqa: E402

SH = -0.40          # every real run so far executed at this shift
RAW = "logs/real_raw_clouds"

# run dir -> (raw-cloud prefix, deployed policy tag)
RUNS = [
    ("logs/crane_policy_debug/run_20260803_155559", "double",     "BASELINE"),
    ("logs/crane_policy_debug/run_20260803_174017", "jagged",     "BASELINE"),
    ("logs/crane_policy_debug/run_20260803_204034", "single",     "BASELINE"),
    ("logs/bc_pointcloud/scoring_margin05_2048_c/policy_debug/run_20260805_153036",
     "p2csingle", "SCORING"),
    ("logs/bc_pointcloud/scoring_margin05_2048_c/policy_debug/run_20260805_194008",
     "p2cdouble", "SCORING"),
]
COL = {"BASELINE": "#1ec91e", "REGRESSION": "#00a6d9", "SCORING": "#d90d0d"}


def q16(p):
    mid = 0.5 * (B_MIN + B_MAX)
    return base64.b64encode(np.clip((p - mid) * 4000.0, -32000, 32000)
                            .astype(np.int16).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/three_policy_3d.html")
    a = ap.parse_args()

    scoring = ScoringGraspPolicy(num_points=2048)
    scoring.load_state_dict(torch.load(
        "logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt",
        map_location="cpu", weights_only=False)["model_state_dict"])
    scoring.eval()
    reg = RegressionBC()
    reg.load_state_dict(torch.load(
        "logs/bc_pointcloud/bc_aug1v2_dig25/bc_pointcloud_policy.pt",
        map_location="cpu", weights_only=False)["model_state_dict"])
    reg.eval()
    heur = HeuristicPolicy(B_MIN, B_MAX, cossin=True, dig=0.30)

    slides = []
    for run, prefix, deployed in RUNS:
        try:
            dec = {json.loads(l)["cycle"]: json.loads(l)
                   for l in open(f"{run}/decisions.jsonl") if json.loads(l)["kind"] == "gaze"}
        except FileNotFoundError:
            continue
        for f in sorted(glob.glob(os.path.join(RAW, f"{prefix}_c*.npz"))):
            cyc = int(os.path.basename(f).rsplit("_c", 1)[1][:-4])
            if cyc not in dec:
                continue
            raw = np.load(f)["points"].astype(np.float32).copy()
            raw[:, 1] -= SH                                    # -> training coords

            # EXACT policy input when the node stored it: the bag's nearest frame is the right
            # SCENE but its own crop+FPS draw differs, and FPS resampling alone can move an
            # argmax pick metres (measured: replay-on-bag y=5.13 vs executed 2.85 on a cloud
            # whose timestamps match to 0.0 s). Only the node's stored cloud reproduces the
            # decision; the bag reconstruction is a scene reference, not a decision reference.
            node_npz = os.path.join(run, f"policy_debug_{cyc:03d}.npz")
            node_pts = None
            if os.path.exists(node_npz):
                _z = np.load(node_npz)
                node_pts = _z["points"].astype(np.float32).copy()
                node_pts[:, 1] -= float(_z["rack_y_shift"])
            src = node_pts if node_pts is not None else raw
            tight = fps(crop(src, 0.0), 1024)                  # baseline + regression input
            marg = (pad_to(src, 2048) if node_pts is not None and len(src) <= 2048
                    else fps(crop(src, 0.5), 2048))            # scoring input
            tg = {}
            with torch.no_grad():
                x, y, z, yaw = heur.get_target(torch.from_numpy(tight).reshape(-1))
                tg["BASELINE"] = [round(float(v), 3) for v in (x, y, z, yaw)]
                tg["REGRESSION"] = [round(float(v), 3) for v in
                                    dec5(reg(torch.from_numpy(pad_to(tight, 2048))[None])[0].numpy())]
                t = scoring.act(torch.from_numpy(pad_to(marg, 2048))[None])[0].numpy()
                tg["SCORING"] = [round(float(v), 3) for v in t]

            ex = dec[cyc]["target"]
            executed = [round(float(ex[0]), 3), round(float(ex[1] - SH), 3),
                        round(float(ex[2]), 3), round(float(ex[3]), 3)]
            disp = crop(raw, 0.5)
            disp = disp[np.random.default_rng(0).choice(
                len(disp), min(len(disp), 9000), replace=False)] if len(disp) > 9000 else disp
            slides.append({
                "t": f"{prefix} cycle {cyc}  -  DEPLOYED: {deployed}"
                     + ("" if node_pts is not None else "   [BAG-CLOUD REPLAY - decision not reproducible]"),
                "dep": deployed, "n": len(disp), "d": q16(disp),
                "tg": tg, "ex": executed,
                "c": "  |  ".join(f"{k} y={v[1]:.2f} z={v[2]:.2f}" for k, v in tg.items())
                     + f"  |  executed y={executed[1]:.2f} z={executed[2]:.2f}"})
        print("done", prefix)

    html = ("""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Baseline vs Regression BC vs Scoring BC - real clouds</title>
<style>body{font-family:sans-serif;margin:10px;background:#111;color:#ddd}
canvas{background:#181818;border:1px solid #333}
.leg span{padding:2px 8px;margin-right:8px;border-radius:3px;color:#fff}
#cap{font-family:monospace;margin:4px 0;color:#aaa;font-size:12px}
input[type=range]{width:100%}</style></head><body>
<b>Drag = orbit &nbsp; wheel = zoom &nbsp; right-drag = pan &nbsp; dbl-click = reset &nbsp; arrows = slide</b>
<div class="leg" style="margin:6px 0">
<span style="background:#1ec91e;color:#000">BASELINE heuristic (tight/1024)</span>
<span style="background:#00a6d9">REGRESSION BC (tight/1024)</span>
<span style="background:#d90d0d">SCORING BC / P2c (margin/2048)</span>
<span style="background:#fff;color:#000">WHITE = the target actually EXECUTED</span>
<span style="background:#555">coloured = replays of each policy on the same cloud</span></div>
<div id="ttl"></div><div id="cap"></div>
<input id="sl" type="range" min="0" max="NMAX" value="0">
<canvas id="cv" width="1200" height="760"></canvas>
<script>
const S=PAYLOAD, MID=MIDV, BMIN=BMINV, BMAX=BMAXV;
const COLS={BASELINE:'#1ec91e',REGRESSION:'#00a6d9',SCORING:'#d90d0d'};
const cv=document.getElementById('cv'),cx=cv.getContext('2d');
let yaw=-0.9,pitch=0.5,zoom=130,panx=0,pany=0,cur=0,pts=null;
function b64i16(s){const b=atob(s);const dv=new DataView(new ArrayBuffer(b.length));
 for(let i=0;i<b.length;i++)dv.setUint8(i,b.charCodeAt(i));
 const a=new Int16Array(b.length/2);for(let i=0;i<a.length;i++)a[i]=dv.getInt16(2*i,true);return a;}
function load(i){cur=i;pts=b64i16(S[i].d);
 document.getElementById('ttl').innerHTML='<b>'+(i+1)+'/'+S.length+'</b>  '+S[i].t+'  ('+S[i].n+' pts shown)';
 document.getElementById('cap').textContent=S[i].c;draw();}
function proj(x,y,z){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 const rx=cy*x-sy*y,ry=sy*x+cy*y;const vz=cp*ry+sp*z,vy=-sp*ry+cp*z;
 return [rx*zoom+cv.width/2+panx,-vy*zoom+cv.height/2+pany,vz];}
function turbo(t){t=Math.max(0,Math.min(1,t));
 const r=Math.max(0,Math.min(1,1.5-Math.abs(4*t-3)))*255,g=Math.max(0,Math.min(1,1.5-Math.abs(4*t-2)))*255,
 b=Math.max(0,Math.min(1,1.5-Math.abs(4*t-1)))*255;return 'rgb('+(r|0)+','+(g|0)+','+(b|0)+')';}
function boxEdges(mn,mx){const c=[[mn[0],mn[1],mn[2]],[mx[0],mn[1],mn[2]],[mx[0],mx[1],mn[2]],
 [mn[0],mx[1],mn[2]],[mn[0],mn[1],mx[2]],[mx[0],mn[1],mx[2]],[mx[0],mx[1],mx[2]],[mn[0],mx[1],mx[2]]];
 const e=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
 return e.map(([i,j])=>[c[i],c[j]]);}
function drawBox(mn,mx,col){cx.strokeStyle=col;cx.lineWidth=1.2;
 for(const [a,b] of boxEdges(mn,mx)){
  const p=proj(a[0]-MID[0],a[1]-MID[1],a[2]-MID[2]),q=proj(b[0]-MID[0],b[1]-MID[1],b[2]-MID[2]);
  cx.beginPath();cx.moveTo(p[0],p[1]);cx.lineTo(q[0],q[1]);cx.stroke();}}
function mark(t,col,isDep,lbl){const p=proj(t[0]-MID[0],t[1]-MID[1],t[2]-MID[2]);
 cx.setLineDash([3,3]);cx.strokeStyle=col;cx.lineWidth=1.5;
 const g=proj(t[0]-MID[0],t[1]-MID[1],BMIN[2]-MID[2]);
 cx.beginPath();cx.moveTo(p[0],p[1]);cx.lineTo(g[0],g[1]);cx.stroke();cx.setLineDash([]);
 cx.strokeStyle=col;cx.lineWidth=2.5;
 const dx=0.55*Math.cos(t[3]),dy=0.55*Math.sin(t[3]);
 const q1=proj(t[0]+dx-MID[0],t[1]+dy-MID[1],t[2]-MID[2]),q2=proj(t[0]-dx-MID[0],t[1]-dy-MID[1],t[2]-MID[2]);
 cx.beginPath();cx.moveTo(q1[0],q1[1]);cx.lineTo(q2[0],q2[1]);cx.stroke();
 if(isDep){cx.strokeStyle='#fff';cx.lineWidth=3;cx.beginPath();cx.arc(p[0],p[1],12,0,7);cx.stroke();}
 cx.fillStyle=col;cx.beginPath();cx.arc(p[0],p[1],8,0,7);cx.fill();
 cx.fillStyle='#fff';cx.font='bold 11px sans-serif';cx.fillText(lbl,p[0]+11,p[1]-6);}
function draw(){cx.clearRect(0,0,cv.width,cv.height);
 const n=S[cur].n,order=[],P=new Array(n);
 for(let i=0;i<n;i++){const x=pts[3*i]/4000,y=pts[3*i+1]/4000,z=pts[3*i+2]/4000;
  P[i]=[proj(x,y,z),z+MID[2]];order.push(i);}
 order.sort((a,b)=>P[a][0][2]-P[b][0][2]);
 for(const i of order){cx.fillStyle=turbo((P[i][1]+1.5)/1.1);
  cx.fillRect(P[i][0][0]-1.2,P[i][0][1]-1.2,2.4,2.4);}
 drawBox(BMIN,BMAX,'#5588cc');
 drawBox([BMIN[0]-0.5,BMIN[1]-0.5,BMIN[2]-0.5],[BMAX[0]+0.5,BMAX[1]+0.5,BMAX[2]],'#cc8822');
 const dep=S[cur].dep;
 for(const k of ['REGRESSION','BASELINE','SCORING'])
   mark(S[cur].tg[k],COLS[k],false,k[0]+k.slice(1,4).toLowerCase());
 mark(S[cur].ex,'#ffffff',true,'EXECUTED ('+dep+')');}
let drag=0,lx=0,ly=0;
cv.onmousedown=e=>{drag=e.button===2?2:1;lx=e.clientX;ly=e.clientY;};
window.onmouseup=()=>drag=0;
window.onmousemove=e=>{if(!drag)return;const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;
 if(drag===1){yaw+=dx*0.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.008));}
 else{panx+=dx;pany+=dy;}draw();};
cv.oncontextmenu=e=>e.preventDefault();
cv.onwheel=e=>{e.preventDefault();zoom*=e.deltaY<0?1.12:0.89;zoom=Math.max(25,Math.min(900,zoom));draw();};
cv.ondblclick=()=>{yaw=-0.9;pitch=0.5;zoom=130;panx=0;pany=0;draw();};
const sl=document.getElementById('sl');sl.oninput=e=>load(+e.target.value);
document.onkeydown=e=>{if(e.key=='ArrowRight'){sl.value=Math.min(+sl.value+1,NMAX);load(+sl.value);}
 if(e.key=='ArrowLeft'){sl.value=Math.max(+sl.value-1,0);load(+sl.value);}};
load(0);
</script></body></html>"""
            .replace("NMAX", str(len(slides) - 1))
            .replace("PAYLOAD", json.dumps(slides))
            .replace("MIDV", json.dumps([round(float(v), 4) for v in 0.5 * (B_MIN + B_MAX)]))
            .replace("BMINV", json.dumps([float(v) for v in B_MIN]))
            .replace("BMAXV", json.dumps([float(v) for v in B_MAX])))
    with open(a.out, "w") as fo:
        fo.write(html)
    print(f"wrote {a.out} ({len(slides)} slides, {os.path.getsize(a.out)//1024} KB)")


if __name__ == "__main__":
    main()
