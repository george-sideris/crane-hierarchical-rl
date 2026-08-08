#!/usr/bin/env python3
"""Interactive 3D version of the policy target comparison: real point clouds you can orbit
(drag), zoom (wheel) and pan (right-drag), with P1/scoring_v1/P2 targets overlaid. Entirely
self-contained HTML - a hand-rolled canvas renderer, no external libraries, works offline.

Clouds are the actual POLICY INPUTS (recorded 1024-pt tight clouds for the gap run; FPS'd
2048-pt margin clouds for the raw trial set), quantized to int16 (~12 KB/slide).

    python3 scripts/envs/build_interactive_comparison.py --out docs/figures/policy_comparison_3d.html
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys

import json as _json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring_head import ScoringGraspPolicy  # noqa: E402
from render_policy_comparison import (RegressionBC, dec5, fps, crop, pad_to,  # noqa: E402
                                      B_MIN, B_MAX, SH, RECORDED, RAW)


def q16(p):
    """quantize a cloud to int16 around the action-box center (mm-scale precision)."""
    mid = 0.5 * (B_MIN + B_MAX)
    q = np.clip((p - mid) * 4000.0, -32000, 32000).astype(np.int16)
    return base64.b64encode(q.tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/policy_comparison_3d.html")
    a = ap.parse_args()

    v1 = ScoringGraspPolicy(num_points=1024)
    v1.load_state_dict(torch.load("logs/bc_pointcloud/scoring_v1/scoring_policy.pt",
                                  map_location="cpu", weights_only=False)["model_state_dict"])
    p2 = ScoringGraspPolicy(num_points=2048)
    p2.load_state_dict(torch.load("logs/bc_pointcloud/scoring_margin05_2048/scoring_policy.pt",
                                  map_location="cpu", weights_only=False)["model_state_dict"])
    p2b = None
    p2b_path = "logs/bc_pointcloud/scoring_margin05_2048_stub/scoring_policy.pt"
    if os.path.exists(p2b_path):
        p2b = ScoringGraspPolicy(num_points=2048)
        p2b.load_state_dict(torch.load(p2b_path, map_location="cpu",
                                       weights_only=False)["model_state_dict"])
        p2b.eval()
    p2c = None
    p2c_path = "logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt"
    _c_done = os.path.exists("logs/train_scoring_c.log") and \
        "best val" in open("logs/train_scoring_c.log").read()
    if os.path.exists(p2c_path) and _c_done:   # only the FINISHED P2c, not a mid-training snapshot
        p2c = ScoringGraspPolicy(num_points=2048)
        p2c.load_state_dict(torch.load(p2c_path, map_location="cpu",
                                       weights_only=False)["model_state_dict"])
        p2c.eval()
    p1 = RegressionBC()
    p1.load_state_dict(torch.load("logs/bc_pointcloud/bc_margin05_2048_reg/bc_pointcloud_policy.pt",
                                  map_location="cpu", weights_only=False)["model_state_dict"])
    for m in (v1, p2, p1):
        m.eval()

    def targets_for(tight_cloud, margin_cloud):
        mt = torch.from_numpy(pad_to(margin_cloud, 2048))[None]
        with torch.no_grad():
            tv = v1.act(torch.from_numpy(pad_to(tight_cloud, 1024))[None])[0].numpy()
            tp2 = p2.act(mt)[0].numpy()
            tp1 = dec5(p1(mt)[0].numpy())
            out = {"v1": [round(float(x), 3) for x in tv],
                   "P2": [round(float(x), 3) for x in tp2],
                   "P1": [round(float(x), 3) for x in tp1]}
            if p2b is not None:
                out["P2b"] = [round(float(x), 3) for x in p2b.act(mt)[0].numpy()]
            if p2c is not None:
                out["P2c"] = [round(float(x), 3) for x in p2c.act(mt)[0].numpy()]
        return out

    RUNS = {"double": "logs/crane_policy_debug/run_20260803_155559",
            "jagged": "logs/crane_policy_debug/run_20260803_174017",
            "single": "logs/crane_policy_debug/run_20260803_204034",
            # today's P2c trial: REAL = what P2c itself executed (green vs its own replay
            # shows cloud-timing sensitivity; run used rack_y_shift -0.4)
            "p2csingle": "logs/bc_pointcloud/scoring_margin05_2048_c/policy_debug/run_20260805_153036",
            "p2cdouble": "logs/bc_pointcloud/scoring_margin05_2048_c/policy_debug/run_20260805_194008"}
    real_tgt = {}
    for tag, run in RUNS.items():
        for l in open(f"{run}/decisions.jsonl"):
            rec = _json.loads(l)
            if rec["kind"] == "gaze":
                real_tgt[(tag, rec["cycle"])] = rec["target"]

    slides = []
    for f in sorted(glob.glob(os.path.join(RECORDED, "policy_debug_*.npz"))):
        d = np.load(f)
        pts = d["points"].astype(np.float32).copy(); pts[:, 1] -= float(d["rack_y_shift"])
        tg = targets_for(pts, pts)
        # the EXECUTED deployed-BC target that cycle (npz stores it in physical coords;
        # same shift as the cloud puts it in the displayed frame)
        r = d["target"].astype(np.float32)
        tg["REAL"] = [round(float(r[0]), 3), round(float(r[1] - d["rack_y_shift"]), 3),
                      round(float(r[2]), 3), round(float(r[3]), 3)]
        slides.append({"t": f"gap-run {os.path.basename(f)[13:16]} (recorded, tight; REAL = deployed BC)",
                       "n": len(pts), "d": q16(pts), "m": 0.0, "tg": tg})
    for f in sorted(glob.glob(os.path.join(RAW, "*.npz"))):
        p = np.load(f)["points"].astype(np.float32).copy(); p[:, 1] -= SH
        tight = fps(crop(p, 0.0), 1024)
        marg = fps(crop(p, 0.5), 2048)
        mv = marg[np.abs(marg).sum(1) > 1e-6]
        tg = targets_for(tight, marg)
        base = os.path.basename(f)[:-4]
        tag, cyc = base.rsplit("_c", 1)
        rt = real_tgt.get((tag, int(cyc)))
        if rt is not None:
            # decisions store the executed PHYSICAL target; the displayed cloud frame is
            # physical shifted by -SH, so apply the same shift (NOT that run's rack_y_shift,
            # which only matters for reproducing the node crop)
            tg["REAL"] = [round(rt[0], 3), round(rt[1] - SH, 3),
                          round(rt[2], 3), round(rt[3], 3)]
        slides.append({"t": f"trial {base} (raw, margin input; REAL = heuristic baseline)",
                       "n": len(mv), "d": q16(mv), "m": 0.5, "tg": tg})
        print("done", f)

    payload = json.dumps(slides)
    mid = [round(float(v), 4) for v in (0.5 * (B_MIN + B_MAX))]
    html = ("""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>P1 / scoring_v1 / P2 - interactive real-cloud targets</title>
<style>body{font-family:sans-serif;margin:10px;background:#111;color:#ddd}
canvas{background:#181818;border:1px solid #333;touch-action:none}
.leg span{padding:2px 8px;margin-right:8px;color:#fff;border-radius:3px}
#cap{font-family:monospace;margin:4px 0;color:#aaa}
input[type=range]{width:100%}</style></head><body>
<b>Drag = orbit &nbsp; wheel = zoom &nbsp; right-drag = pan &nbsp; dbl-click = reset &nbsp; arrows = slide</b>
<div class="leg" style="margin:6px 0"><span style="background:#d90d0d">P2 scoring+margin</span>
<span style="background:#8a2be2">P2b stub-variation</span>
<span style="background:#f2a900;color:#000">P2c stub-negatives</span>
<span style="background:#d90dbf">scoring_v1 tight</span>
<span style="background:#00a6d9">P1 regression+margin</span>
<span style="background:#1ec91e;color:#000">REAL executed (BC / heuristic)</span>
<span style="background:#345">action box</span><span style="background:#a70">margin box</span></div>
<div id="ttl"></div><div id="cap"></div>
<input id="sl" type="range" min="0" max="NMAX" value="0">
<canvas id="cv" width="1200" height="760"></canvas>
<script>
const S=PAYLOAD, MID=MIDV, BMIN=BMINV, BMAX=BMAXV;
const cv=document.getElementById('cv'),cx=cv.getContext('2d');
let yaw=-0.9,pitch=0.5,zoom=130,panx=0,pany=0,cur=0,pts=null;
function b64i16(s){const b=atob(s);const a=new Int16Array(b.length/2);
 const dv=new DataView(new ArrayBuffer(b.length));
 for(let i=0;i<b.length;i++)dv.setUint8(i,b.charCodeAt(i));
 for(let i=0;i<a.length;i++)a[i]=dv.getInt16(2*i,true);return a;}
function load(i){cur=i;pts=b64i16(S[i].d);
 document.getElementById('ttl').innerHTML='<b>'+(i+1)+'/'+S.length+'</b>  '+S[i].t+'  ('+S[i].n+' pts)';
 const tg=S[i].tg;document.getElementById('cap').textContent=
  Object.entries(tg).map(([k,v])=>k+' y='+v[1].toFixed(2)+' z='+v[2].toFixed(2)).join('   |   ');
 draw();}
function proj(x,y,z){ // world (m, centered) -> rotated view
 const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 const rx=cy*x - sy*y, ry=sy*x + cy*y;
 const vz=cp*ry + sp*z, vy=-sp*ry + cp*z;   // vz = depth-ish, vy = up
 return [rx*zoom + cv.width/2 + panx, -vy*zoom + cv.height/2 + pany, vz];}
function turbo(t){t=Math.max(0,Math.min(1,t));
 const r=Math.max(0,Math.min(1,1.5-Math.abs(4*t-3)))*255,
       g=Math.max(0,Math.min(1,1.5-Math.abs(4*t-2)))*255,
       b=Math.max(0,Math.min(1,1.5-Math.abs(4*t-1)))*255;
 return 'rgb('+(r|0)+','+(g|0)+','+(b|0)+')';}
function boxEdges(mn,mx){const c=[[mn[0],mn[1],mn[2]],[mx[0],mn[1],mn[2]],[mx[0],mx[1],mn[2]],
 [mn[0],mx[1],mn[2]],[mn[0],mn[1],mx[2]],[mx[0],mn[1],mx[2]],[mx[0],mx[1],mx[2]],[mn[0],mx[1],mx[2]]];
 const e=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
 return e.map(([i,j])=>[c[i],c[j]]);}
function drawBox(mn,mx,col){cx.strokeStyle=col;cx.lineWidth=1.2;
 for(const [a,b] of boxEdges(mn,mx)){
  const p=proj(a[0]-MID[0],a[1]-MID[1],a[2]-MID[2]),q=proj(b[0]-MID[0],b[1]-MID[1],b[2]-MID[2]);
  cx.beginPath();cx.moveTo(p[0],p[1]);cx.lineTo(q[0],q[1]);cx.stroke();}}
function draw(){cx.clearRect(0,0,cv.width,cv.height);
 const s=S[cur];const n=s.n;const order=[];
 const P=new Array(n);
 for(let i=0;i<n;i++){const x=pts[3*i]/4000,y=pts[3*i+1]/4000,z=pts[3*i+2]/4000;
  P[i]=[proj(x,y,z),z+MID[2]];order.push(i);}
 order.sort((a,b)=>P[a][0][2]-P[b][0][2]);           // painter: far first
 for(const i of order){const [[px,py],zz]=[[P[i][0][0],P[i][0][1]],P[i][1]];
  cx.fillStyle=turbo((zz+1.5)/1.1);cx.fillRect(px-1.5,py-1.5,3,3);}
 drawBox(BMIN,BMAX,'#5588cc');
 if(s.m>0)drawBox([BMIN[0]-s.m,BMIN[1]-s.m,BMIN[2]-s.m],[BMAX[0]+s.m,BMAX[1]+s.m,BMAX[2]],'#cc8822');
 const cols={P2:'#d90d0d',P2b:'#8a2be2',P2c:'#f2a900',v1:'#d90dbf',P1:'#00a6d9',REAL:'#1ec91e'};
 for(const [k,t] of Object.entries(s.tg)){const col=cols[k];
  const x=t[0]-MID[0],y=t[1]-MID[1],z=t[2]-MID[2];
  const p=proj(x,y,z);
  cx.strokeStyle=col;cx.lineWidth=2.5;                       // yaw line
  const dx=0.55*Math.cos(t[3]),dy=0.55*Math.sin(t[3]);
  const q1=proj(x+dx,y+dy,z),q2=proj(x-dx,y-dy,z);
  cx.beginPath();cx.moveTo(q1[0],q1[1]);cx.lineTo(q2[0],q2[1]);cx.stroke();
  cx.beginPath();cx.setLineDash([3,3]);                      // drop line for depth cue
  const g=proj(x,y,BMIN[2]-MID[2]);
  cx.moveTo(p[0],p[1]);cx.lineTo(g[0],g[1]);cx.stroke();cx.setLineDash([]);
  cx.fillStyle=col;cx.beginPath();cx.arc(p[0],p[1],k=='REAL'?9:7,0,7);cx.fill();
  cx.fillStyle='#fff';cx.font='bold 11px sans-serif';cx.fillText(k,p[0]+9,p[1]-6);}}
let drag=0,lx=0,ly=0;
cv.onmousedown=e=>{drag=e.button===2?2:1;lx=e.clientX;ly=e.clientY;};
window.onmouseup=()=>drag=0;
window.onmousemove=e=>{if(!drag)return;const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;
 if(drag===1){yaw+=dx*0.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.008));}
 else{panx+=dx;pany+=dy;}draw();};
cv.oncontextmenu=e=>e.preventDefault();
cv.onwheel=e=>{e.preventDefault();zoom*=e.deltaY<0?1.12:0.89;zoom=Math.max(25,Math.min(900,zoom));draw();};
cv.ondblclick=()=>{yaw=-0.9;pitch=0.5;zoom=130;panx=0;pany=0;draw();};
const sl=document.getElementById('sl');
sl.oninput=e=>load(+e.target.value);
document.onkeydown=e=>{if(e.key=='ArrowRight'){sl.value=Math.min(+sl.value+1,NMAX);load(+sl.value);}
 if(e.key=='ArrowLeft'){sl.value=Math.max(+sl.value-1,0);load(+sl.value);}};
load(0);
</script></body></html>"""
            .replace("NMAX", str(len(slides) - 1))
            .replace("PAYLOAD", payload)
            .replace("MIDV", json.dumps(mid))
            .replace("BMINV", json.dumps([float(v) for v in B_MIN]))
            .replace("BMAXV", json.dumps([float(v) for v in B_MAX])))
    with open(a.out, "w") as fo:
        fo.write(html)
    print(f"wrote {a.out} ({len(slides)} slides, {os.path.getsize(a.out)//1024} KB)")


if __name__ == "__main__":
    main()
