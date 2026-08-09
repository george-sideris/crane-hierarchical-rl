#!/usr/bin/env python3
"""Convert an RSL-RL scoring checkpoint (model_N.pt) into the format play_bc_pointcloud reads.

Only two things differ:

  keys      ScoringActorCritic carries `critic.*`, `log_sigma_dz`, `log_sigma_yaw` on top of the
            actor. ScoringGraspPolicy has ONLY the actor (encoder / head / score_out / dz_out /
            yaw_out) and load_state_dict is strict, so the extras must be dropped.
  metadata  The eval reads `num_points`, `dig`, `bounds_min/max` from the checkpoint. RSL-RL
            never stores them, so they are copied from the BC checkpoint the run was
            initialised from (--template). Getting num_points wrong silently changes the cloud
            pipeline, so the template is REQUIRED rather than guessed.

The counterpart `convert_bc_to_rsl_rl.py` goes the other way (BC -> training init). The
`*_bc_format.pt` files in logs/ were made by an uncommitted version of this.

    python3 scripts/envs/rsl_rl_to_scoring.py \
        --rsl_rl logs/rsl_rl/crane_scoring_ppo_v2/<run>/model_150.pt \
        --template logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt
"""

from __future__ import annotations

import argparse
import os

import torch

DROP_PREFIXES = ("critic.",)
DROP_KEYS = ("log_sigma_dz", "log_sigma_yaw", "std")
CARRY = ("num_points", "arch", "dig", "bounds_min", "bounds_max")


def convert(rsl_rl_path, template_path, out_path=None):
    src = torch.load(rsl_rl_path, map_location="cpu", weights_only=False)
    sd = src.get("model_state_dict", src)
    tpl = torch.load(template_path, map_location="cpu", weights_only=False)

    actor = {k: v for k, v in sd.items()
             if not k.startswith(DROP_PREFIXES) and k not in DROP_KEYS}
    dropped = sorted(set(sd) - set(actor))

    # Fail loudly on a key mismatch: a silently half-loaded actor evaluates as noise and the
    # resulting row would look like "RL made it worse" rather than "the conversion was wrong".
    expect = set(tpl["model_state_dict"])
    missing, extra = expect - set(actor), set(actor) - expect
    if missing or extra:
        raise SystemExit(f"key mismatch vs template\n  missing: {sorted(missing)[:6]}\n"
                         f"  extra:   {sorted(extra)[:6]}")

    out = {"model_state_dict": actor}
    for k in CARRY:
        if k in tpl:
            out[k] = tpl[k]
    out["metadata"] = {**tpl.get("metadata", {}),
                       "converted_from": os.path.abspath(rsl_rl_path),
                       "rsl_rl_iter": int(src.get("iter", -1)),
                       "template": os.path.abspath(template_path)}
    if out_path is None:
        out_path = rsl_rl_path.replace(".pt", "_bc_format.pt")
    torch.save(out, out_path)
    print(f"wrote {out_path}  (iter {out['metadata']['rsl_rl_iter']}, "
          f"num_points {out.get('num_points')}, dropped {len(dropped)} non-actor keys)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsl_rl", required=True)
    ap.add_argument("--template", required=True,
                    help="BC scoring checkpoint to copy num_points/dig/bounds from")
    ap.add_argument("--output", default=None)
    a = ap.parse_args()
    convert(a.rsl_rl, a.template, a.output)


if __name__ == "__main__":
    main()
