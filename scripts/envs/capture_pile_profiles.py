#!/usr/bin/env python3
"""Capture one settled pile per height profile, for the pile-variation figure of Chapter 3.

The chapter enumerates the profile families the spawner draws from but shows none of them,
and the families cannot be recovered after the fact: the carved shape relaxes as the logs
settle, so classifying a recorded cloud by its shape guesses at the draw instead of reading
it. Here the profile is pinned with --force_pile_profile, which the spawner reads at spawn
time, so one session can walk the families and each capture is labelled by construction.

Never steps the environment. A step is a whole grasp cycle in this env, which would empty
part of the rack before the pile is photographed; reset settles the pile on its own and the
camera is pumped by rendering.

Runs INSIDE the container:
    ./isaaclab.sh -p scripts/envs/capture_pile_profiles.py --profiles flat ramp_far two_mounds
"""

import argparse
import os
import sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--profiles", nargs="+",
                 default=["flat", "ramp_far", "two_mounds"],
                 help="subset of the spawner's height profiles to capture")
_ap.add_argument("--out", default="/workspace/crane_testbed/logs/pile_profiles")
_ap.add_argument("--num_logs", type=int, default=200)
_ap.add_argument("--seed", type=int, default=7)
_ap.add_argument("--settle_renders", type=int, default=60)
_ap.add_argument("--converge_renders", type=int, default=400,
                 help="renders after hiding the crane, so the accumulated image loses "
                      "its ghost before the overview frame is read")
_ap.add_argument("--rack_usd", type=str, default=None,
                 help="override the rack asset, for testing collision authoring")
_ap.add_argument("--ang_damping", type=float, default=3.0,
                 help="angular damping for the logs. Friction stops sliding but a "
                      "cylinder rolls down any slope regardless; this is the knob that "
                      "stops rolling, so raise it hard to make the logs settle straight down")
_ap.add_argument("--peak", type=float, default=None,
                 help="pass through to the env as --pile_peak_center")
_ap.add_argument("--centers", type=float, nargs=2, default=None,
                 help="pass through to the env as --pile_mound_centers")
_ap.add_argument("--width", type=float, default=None,
                 help="pass through to the env as --pile_mound_width")
_ap.add_argument("--repose", type=float, nargs=2, default=None,
                 help="pass through to the env as --pile_repose_deg")
_ap.add_argument("--amp", type=float, nargs=2, default=None,
                 help="pass through to the env as --pile_amp")
_ap.add_argument("--log_friction", type=float, default=0.0,
                 help="static friction for the logs. Left at the env default, which is what "
                      "collection and evaluation run with, so the captured piles are the ones "
                      "the policies actually see. Raising it to 0.9 was measured to change the "
                      "settled relief by a few centimetres: the profile is slope-limited when it "
                      "is carved, so settling is not what flattens it")
_ap.add_argument("--hide_crane", action="store_true", default=True,
                 help="hide the machine for the overview frame so the pile crest is not\n                      crossed by the trailer deck")
_ap.add_argument("--focal", type=float, default=14.0,
                 help="overview camera focal length in mm; its default of 5.0 is a wide angle "
                      "that leaves the pile small against a lot of empty floor")
_ap.add_argument("--eye_offset", type=float, nargs=3, default=[-11.0, 0.0, -0.15],
                 help="overview camera eye, as an offset from the action-box centre in the base "
                      "frame. The default looks along +x, across the rack rather than down its "
                      "length, so the height profile shows as an elevation")
_ap.add_argument("--target_offset", type=float, nargs=3, default=[0.0, 0.0, -0.15],
                 help="what the camera looks at, as an offset from the action-box centre. Matching "
                      "its height to the eye gives a level, straight-on elevation; aiming at the "
                      "box centre from above tilts the view and foreshortens the relief")
_mine, _rest = _ap.parse_known_args()

# The env module parses the real command line when imported, so hand it the flags the pile
# spawner needs before importing it. Its own --seed/--num_logs live on that parser.
sys.argv = [sys.argv[0], "--headless", "--num_envs", "1", "--profile_piles",
            "--num_logs", str(_mine.num_logs), "--seed", str(_mine.seed),
            "--log_scale_mean", "1.0", "--log_scale_jitter", "0.10",
            "--log_ang_damping", str(_mine.ang_damping), "--gripper_effort", "2000",
            "--log_friction", str(_mine.log_friction)]
if _mine.repose:
    sys.argv += ["--pile_repose_deg", str(_mine.repose[0]), str(_mine.repose[1])]
if _mine.amp:
    sys.argv += ["--pile_amp", str(_mine.amp[0]), str(_mine.amp[1])]
if _mine.rack_usd:
    sys.argv += ["--rack_usd", _mine.rack_usd]
if _mine.peak is not None:
    sys.argv += ["--pile_peak_center", str(_mine.peak)]
if _mine.centers:
    sys.argv += ["--pile_mound_centers", str(_mine.centers[0]), str(_mine.centers[1])]
if _mine.width:
    sys.argv += ["--pile_mound_width", str(_mine.width)]

from isaaclab.app import AppLauncher                                    # noqa: E402

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402
import crane_rl_env_gaze as G                                           # noqa: E402
from crane_rl_env_gaze import CraneDirectEnvFull, CraneDirectEnvCfgFull  # noqa: E402

BOX_MIN = (-5.364, -1.684, -1.30)
BOX_MAX = (-3.364, 5.316, 0.10)


def _to_world(env, p_base):
    """Base-frame point to world, using the crane root pose (the base frame's origin)."""
    pos = env.crane.data.root_pos_w[0].detach().cpu().numpy()
    w, x, y, z = env.crane.data.root_quat_w[0].detach().cpu().numpy()
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])
    return (pos + R @ np.asarray(p_base, dtype=float)).tolist()


def main():
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.use_hierarchical_rl = True
    cfg.action_space = 5
    cfg.enable_camera = True
    cfg.camera_cfg.width = 1280
    cfg.camera_cfg.height = 720
    cfg.camera_cfg.data_types = ["rgb", "depth", "semantic_segmentation"]
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.enable_stabilization = True
    # Creates the world-space overview camera. The basemast camera is the policy's viewpoint,
    # not a presentation one: it sits low and oblique, so the pile is foreshortened and the
    # crane's own boom takes a third of the frame. The overview camera floats free and can be
    # aimed at the rack. Video writers are opened during stepping, which never happens here.
    cfg.record_video = True
    cfg.overview_camera_cfg.spawn.focal_length = _mine.focal

    env = CraneDirectEnvFull(cfg)
    os.makedirs(_mine.out, exist_ok=True)

    import torch

    for profile in _mine.profiles:
        G.args_cli.force_pile_profile = profile
        env.reset()

        # Reset leaves the arm at its home pose and the gaze pose is applied by the env's
        # control loop, which only runs inside step(). Without this the camera looks at the
        # crane's own boom, the rack sits half out of frame, and the cloud covers a third of
        # the rack. Drive the arm there directly instead of stepping the state machine.
        q = torch.tensor([env.GAZE_SLEW_RAD, env.GAZE_ARM_BOOM_RAD,
                          env.GAZE_ARM_STICK_RAD, env.GAZE_ARM_TELESCOPE_M],
                         device=env.device, dtype=torch.float32)
        q = q.unsqueeze(0).expand(env.num_envs, -1).contiguous()
        env.crane.set_joint_position_target(q, joint_ids=env._ctrl_joint_idx)
        env.crane.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=env._ctrl_joint_idx)
        env.scene.write_data_to_sim()
        for _ in range(240):                      # 2 s at 120 Hz: swing decays, pile holds
            env.sim.step(render=False)
        for _ in range(_mine.settle_renders):
            env.sim.render()
        try:
            env._camera.update(dt=0.0)
        except Exception as exc:
            print("[capture] camera update: %s" % exc)

        rgb = env._camera.data.output["rgb"][0].detach().cpu().numpy()
        pts = env.get_pointcloud_base(0, max_points=20000).detach().cpu().numpy()

        # Where the log BODIES rest, not where the camera sees surfaces. If the rack's collision
        # hull sits proud of its visual bed, the logs stop high and the pile looks like it hovers
        # while the cloud still shows the bed at its rendered height.
        log_pos_b = None
        try:
            lp = env._logs_obj.data.root_pos_w.detach().cpu().numpy()
            base_w = env.crane.data.root_pos_w[0].detach().cpu().numpy()
            log_pos_b = lp - base_w                        # crane base frame (no rotation in z)
            radius = 0.113 * 0.5
            print("[capture] %-11s log centres z: min %.3f p1 %.3f  -> lowest log BOTTOM %.3f" % (
                profile, log_pos_b[:, 2].min(), np.percentile(log_pos_b[:, 2], 1),
                log_pos_b[:, 2].min() - radius))
        except Exception as exc:
            print("[capture] log position probe failed: %s" % exc)

        # Aim the overview camera at the middle of the action box, from the near side and
        # above, so the rack runs across the frame and the pile's relief is side-on.
        # The trailer deck sits behind the rack at almost exactly pile-top height, so in a level
        # elevation it cuts across the crest, which is the line the figure exists to show. The
        # machine is not the subject here, so it is hidden for the overview frame only; the gaze
        # frame above was already captured with it in place.
        crane_prim = None
        if _mine.hide_crane:
            from pxr import UsdGeom
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            crane_prim = stage.GetPrimAtPath("/World/envs/env_0/Crane")
            if crane_prim and crane_prim.IsValid():
                UsdGeom.Imageable(crane_prim).MakeInvisible()
            else:
                print("[capture] crane prim not found, leaving it visible")
                crane_prim = None

        over = None
        if env._overview_camera is not None:
            mid_b = 0.5 * (np.array(BOX_MIN) + np.array(BOX_MAX))
            eye_b = mid_b + np.array(_mine.eye_offset, dtype=float)
            ctr_b = mid_b + np.array(_mine.target_offset, dtype=float)
            eye_w, ctr_w = _to_world(env, eye_b), _to_world(env, ctr_b)
            env._overview_camera.set_world_poses_from_view(
                torch.tensor([eye_w], device=env.device, dtype=torch.float32),
                torch.tensor([ctr_w], device=env.device, dtype=torch.float32))
            # The path tracer accumulates across frames, so renders taken soon after hiding the
            # crane still carry a translucent ghost of it over the rack. 90 frames was not
            # enough to clear it on the captures after the first.
            for _ in range(_mine.converge_renders):
                env.sim.render()
            env._overview_camera.update(dt=0.0)
            over = env._overview_camera.data.output["rgb"][0].detach().cpu().numpy()

        if crane_prim is not None:
            from pxr import UsdGeom
            UsdGeom.Imageable(crane_prim).MakeVisible()

        out = os.path.join(_mine.out, "pile_%s.npz" % profile)
        arrays = dict(rgb=rgb, base_points=pts, profile=profile)
        if log_pos_b is not None:
            arrays["log_pos"] = log_pos_b
        if over is not None:
            arrays["overview"] = over
        np.savez_compressed(out, **arrays)
        print("[capture] %-11s gaze %s  overview %s  cloud %s -> %s" % (
            profile, rgb.shape, None if over is None else over.shape, pts.shape, out))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
