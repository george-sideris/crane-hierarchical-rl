#!/usr/bin/env python3
"""Interactive stub-augmentation preview: screened settled clouds, three slides per cloud
(original / poles truncated / + injected stubs). The camera persists across slides, so
arrow-keys give an A/B/C diff at any orbit angle. Ghost gray = removed points (shown at
their original positions), crimson = injected stub points.

    python3 scripts/envs/build_stub_aug_interactive.py --out docs/figures/stub_aug_3d.html
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stub_aug import truncate_poles, inject_stub  # noqa: E402

B_MIN = np.array([-5.364, -1.684, -1.30], dtype=np.float32)
B_MAX = np.array([-3.364, 5.316, 0.10], dtype=np.float32)
DS = "logs/bc_pointcloud/bc_margin05_2048_500"


def q16(p):
    mid = 0.5 * (B_MIN + B_MAX)
    return base64.b64encode(np.clip((p - mid) * 4000.0, -32000, 32000)
                            .astype(np.int16).tobytes()).decode()


def dec(a):
    xyz = B_MIN + (np.tanh(a[:3]) + 1.0) / 2.0 * (B_MAX - B_MIN)
    return xyz[:2].astype(np.float32)


def screened_picks(pc, n_want=10, rng=None):
    """Representative clouds by PILE substance (pole columns excluded from the counts -
    they reach the crop ceiling in every margin cloud, so max-z says nothing). Mix of
    full (>800 pile pts), mid (300-800) and late (100-300) stages, wide piles preferred."""
    from stub_aug import find_pole_columns
    rng = rng or np.random.default_rng(11)
    buckets = {"full": [], "mid": [], "late": []}
    order = rng.permutation(len(pc))
    for i in order:
        p = np.asarray(pc[i]).astype(np.float32)
        v = p[np.abs(p).sum(-1) > 1e-6]
        polem = np.zeros(len(p), bool)
        for _, m in find_pole_columns(p):
            polem |= m
        vp = p[(np.abs(p).sum(-1) > 1e-6) & ~polem]
        pile = vp[vp[:, 2] > -1.05]
        n = len(pile)
        span = (np.percentile(pile[:, 1], 95) - np.percentile(pile[:, 1], 5)) if n > 50 else 0
        if n > 800 and span > 3.5 and len(buckets["full"]) < 4:
            buckets["full"].append(int(i))
        elif 300 <= n <= 800 and span > 2.0 and len(buckets["mid"]) < 3:
            buckets["mid"].append(int(i))
        elif 100 <= n < 300 and len(buckets["late"]) < 3:
            buckets["late"].append(int(i))
        if sum(len(b) for b in buckets.values()) >= n_want:
            break
    return buckets["full"] + buckets["mid"] + buckets["late"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/stub_aug_3d.html")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()

    pc = np.load(os.path.join(DS, "pointclouds.npy"), mmap_mode="r")
    ac = np.load(os.path.join(DS, "actions.npy"))
    cam = np.load(os.path.join(DS, "cam_pos_base.npy"))
    rng = np.random.default_rng(7)
    picks = screened_picks(pc, a.n)
    print("picks:", picks)

    slides = []
    for i in picks:
        p = np.asarray(pc[i]).astype(np.float32)
        lab = dec(ac[i])
        valid = np.abs(p).sum(-1) > 1e-6

        t, removed = truncate_poles(p, rng)
        s = t.copy()
        inj = np.zeros(len(p), bool)
        for _ in range(2):
            s, m = inject_stub(s, cam, rng, label_xy=lab)
            inj |= m

        # stage A: original, all normal
        pa = p[valid]
        slides.append({"t": f"cloud {i} - 1/3 ORIGINAL", "n": len(pa), "d": q16(pa),
                       "c": base64.b64encode(np.zeros(len(pa), np.uint8).tobytes()).decode()})
        # stage B: kept points normal + removed points ghosted at original positions
        keepm = valid & ~removed
        pb = np.concatenate([p[keepm], p[removed]])
        cb = np.concatenate([np.zeros(keepm.sum(), np.uint8),
                             np.ones(int(removed.sum()), np.uint8)])
        slides.append({"t": f"cloud {i} - 2/3 POLES TRUNCATED (gray = removed)",
                       "n": len(pb), "d": q16(pb),
                       "c": base64.b64encode(cb.tobytes()).decode()})
        # stage C: augmented cloud, injected points crimson, ghosts kept for reference
        sv = np.abs(s).sum(-1) > 1e-6
        pcls = np.zeros(len(s), np.uint8)
        pcls[inj] = 2
        pc_ = np.concatenate([s[sv], p[removed]])
        cc = np.concatenate([pcls[sv], np.ones(int(removed.sum()), np.uint8)])
        slides.append({"t": f"cloud {i} - 3/3 + INJECTED STUBS (crimson)",
                       "n": len(pc_), "d": q16(pc_),
                       "c": base64.b64encode(cc.tobytes()).decode()})

    payload = json.dumps(slides)
    mid = [round(float(v), 4) for v in (0.5 * (B_MIN + B_MAX))]
    html = ("""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Stub augmentation - interactive preview</title>
<style>body{font-family:sans-serif;margin:10px;background:#111;color:#ddd}
canvas{background:#181818;border:1px solid #333;touch-action:none}
.leg span{padding:2px 8px;margin-right:8px;border-radius:3px}
input[type=range]{width:100%}</style></head><body>
<b>Arrow keys flip original / truncated / injected at a fixed camera &nbsp; drag = orbit &nbsp; wheel = zoom &nbsp; right-drag = pan &nbsp; dbl-click = reset</b>
<div class="leg" style="margin:6px 0"><span style="background:#666">removed pole points (ghost)</span>
<span style="background:#c22;color:#fff">injected stub points</span>
<span style="background:#345;color:#fff">action box</span><span style="background:#a70;color:#000">margin box</span></div>
<div id="ttl"></div>
<input id="sl" type="range" min="0" max="NMAX" value="0">
<canvas id="cv" width="1200" height="760"></canvas>
<script>
const S=PAYLOAD, MID=MIDV, BMIN=BMINV, BMAX=BMAXV;
const cv=document.getElementById('cv'),cx=cv.getContext('2d');
let yaw=-0.9,pitch=0.5,zoom=130,panx=0,pany=0,cur=0,pts=null,cls=null;
function b64i16(s){const b=atob(s);const dv=new DataView(new ArrayBuffer(b.length));
 for(let i=0;i<b.length;i++)dv.setUint8(i,b.charCodeAt(i));
 const a=new Int16Array(b.length/2);
 for(let i=0;i<a.length;i++)a[i]=dv.getInt16(2*i,true);return a;}
function b64u8(s){const b=atob(s);const a=new Uint8Array(b.length);
 for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a;}
function load(i){cur=i;pts=b64i16(S[i].d);cls=b64u8(S[i].c);
 document.getElementById('ttl').innerHTML='<b>'+(i+1)+'/'+S.length+'</b>  '+S[i].t+'  ('+S[i].n+' pts)';
 draw();}
function proj(x,y,z){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 const rx=cy*x-sy*y,ry=sy*x+cy*y;const vz=cp*ry+sp*z,vy=-sp*ry+cp*z;
 return [rx*zoom+cv.width/2+panx,-vy*zoom+cv.height/2+pany,vz];}
function turbo(t){t=Math.max(0,Math.min(1,t));
 const r=Math.max(0,Math.min(1,1.5-Math.abs(4*t-3)))*255,g=Math.max(0,Math.min(1,1.5-Math.abs(4*t-2)))*255,
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
 const n=S[cur].n,order=[],P=new Array(n);
 for(let i=0;i<n;i++){const x=pts[3*i]/4000,y=pts[3*i+1]/4000,z=pts[3*i+2]/4000;
  P[i]=[proj(x,y,z),z+MID[2]];order.push(i);}
 order.sort((a,b)=>P[a][0][2]-P[b][0][2]);
 for(const i of order){const px=P[i][0][0],py=P[i][0][1],zz=P[i][1];
  if(cls[i]===1){cx.fillStyle='rgba(150,150,150,0.55)';cx.fillRect(px-1,py-1,2,2);}
  else if(cls[i]===2){cx.fillStyle='#e33';cx.fillRect(px-2,py-2,4,4);}
  else{cx.fillStyle=turbo((zz+1.5)/1.1);cx.fillRect(px-1.5,py-1.5,3,3);}}
 drawBox(BMIN,BMAX,'#5588cc');
 drawBox([BMIN[0]-0.5,BMIN[1]-0.5,BMIN[2]-0.5],[BMAX[0]+0.5,BMAX[1]+0.5,BMAX[2]],'#cc8822');}
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
