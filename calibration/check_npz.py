#!/usr/bin/env python3
"""Quick check: policy target vs cloud content for a policy_debug npz.
Usage: python3 check_npz.py <path-to-policy_debug_XXX.npz> [--plot]"""
import sys, numpy as np
f=sys.argv[1]; z=np.load(f)
pc=z["points"]; t=z["target"]; live=pc[np.abs(pc).sum(1)>0]
near=live[np.linalg.norm(live[:,:2]-t[:2],axis=1)<0.4]
hi=live[live[:,2]>np.percentile(live[:,2],90)]
print(f"target      x={t[0]:.2f} y={t[1]:.2f} z={t[2]:.2f} yaw={t[3]:.2f}")
print(f"cloud       {len(live)} pts  z[{live[:,2].min():.2f},{live[:,2].max():.2f}]  y[{live[:,1].min():.2f},{live[:,1].max():.2f}]")
print(f"near target {len(near)} pts within 0.4m" + (f", top z {near[:,2].max():.2f}" if len(near) else "  <-- EMPTY, no logs at target"))
print(f"log mass    top-10% centroid x={hi[:,0].mean():.2f} y={hi[:,1].mean():.2f} z={hi[:,2].mean():.2f}")
if "--plot" in sys.argv:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(6,7)); plt.scatter(live[:,1],live[:,0],s=3,c=live[:,2],cmap="viridis")
    plt.scatter([t[1]],[t[0]],c="red",marker="x",s=200,label="target")
    plt.xlabel("y (far-right = high)"); plt.ylabel("x"); plt.legend(); plt.gca().invert_xaxis()
    out=f.replace(".npz","_topdown.png"); plt.savefig(out,dpi=90); print("saved",out)
