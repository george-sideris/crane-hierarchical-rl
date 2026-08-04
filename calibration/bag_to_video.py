#!/usr/bin/env python3
"""Extract an RGB image topic from a rosbag2 into an mp4 (HOST, no ROS -- uses the rosbags lib).

Playback fps is taken from the recorded message timestamps (so it plays at real speed),
overridable with --fps. Streams frame-by-frame so big/long bags don't blow up memory.

Usage:
    python3 bag_to_video.py <bag_dir> [--topic /zed_0/zed_node/rgb/color/rect/image]
                                      [--out out.mp4] [--fps N] [--max-frames N]
    python3 bag_to_video.py <bag_dir> --list          # just list image topics and exit
"""
import argparse
import io
import os

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_FOXY)


def decode(msg):
    """sensor_msgs/Image -> HxWx3 uint8 RGB, or None if the encoding is unsupported."""
    h, w, enc = msg.height, msg.width, msg.encoding
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(h, w, 3)
        if enc == "bgr8":
            img = img[:, :, ::-1]
    elif enc in ("rgba8", "bgra8"):
        img = buf.reshape(h, w, 4)[:, :, :3]
        if enc == "bgra8":
            img = img[:, :, ::-1]
    elif enc in ("mono8", "8UC1"):
        img = np.repeat(buf.reshape(h, w, 1), 3, axis=2)
    else:
        return None
    return np.ascontiguousarray(img)


def decode_compressed(msg, swap_rb):
    """sensor_msgs/CompressedImage (jpeg/png) -> HxWx3 uint8 RGB.

    A JPEG produced by compressed_image_transport/cv2 already decodes to correct RGB via a normal
    reader (PIL/imageio), even when the format says bgr8 -- cv2 handled the channel order at encode
    time. So the default is NO swap. swap_rb True/False forces/disables an R<->B swap for the rare
    camera/plugin that stores channels the other way."""
    data = bytes(msg.data)
    try:
        from PIL import Image as PILImage
        img = np.asarray(PILImage.open(io.BytesIO(data)).convert("RGB"))
    except Exception:
        import imageio.v2 as iio
        img = np.asarray(iio.imread(data))
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        img = img[:, :, :3]
    if bool(swap_rb):   # default (None) -> no swap
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def main():
    ap = argparse.ArgumentParser(description="Extract an image topic from a rosbag to mp4")
    ap.add_argument("bag", help="rosbag2 directory (contains metadata.yaml + .db3)")
    ap.add_argument("--topic", default="/zed_0/zed_node/rgb/color/rect/image")
    ap.add_argument("--out", default=None, help="output mp4 (default <bag>_<topic>.mp4 beside the bag)")
    ap.add_argument("--fps", type=float, default=None, help="override fps (default: from timestamps)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--list", action="store_true", help="list topics in the bag and exit")
    ap.add_argument("--swap-rb", dest="swap_rb", action="store_true", default=None,
                    help="force R<->B swap for compressed frames (default: no swap, which is correct "
                         "for cv2/compressed_image_transport jpegs)")
    ap.add_argument("--no-swap-rb", dest="swap_rb", action="store_false",
                    help="force NO R<->B swap for compressed frames (same as the default)")
    args = ap.parse_args()

    bag = args.bag.rstrip("/")

    if args.list:
        with Reader(bag) as r:
            for c in r.connections:
                print(f"  {c.topic}   ({c.msgtype})")
        return

    # Pass 1: collect timestamps (no deserialize) to derive real playback fps.
    print(f"reading timestamps on {args.topic} ...", flush=True)
    with Reader(bag) as r:
        conns = [c for c in r.connections if c.topic == args.topic]
        if not conns:
            print(f"topic {args.topic} not in bag. Available:")
            for c in r.connections:
                print(f"  {c.topic}   ({c.msgtype})")
            raise SystemExit(1)
        stamps = [t for _, t, _ in r.messages(connections=conns)]
    if not stamps:
        raise SystemExit(f"no messages on {args.topic}")
    if args.fps:
        fps = args.fps
    elif len(stamps) > 1:
        # Average fps over the whole bag (robust to per-frame jitter/bursts that fool a median gap).
        dur = (stamps[-1] - stamps[0]) / 1e9
        fps = float((len(stamps) - 1) / dur) if dur > 0 else 15.0
    else:
        fps = 15.0

    out = args.out or (bag + "_" + args.topic.strip("/").replace("/", "_") + ".mp4")

    total = min(len(stamps), args.max_frames) if args.max_frames else len(stamps)
    print(f"decoding {total} frames @ {fps:.1f} fps -> {out} ...", flush=True)

    import imageio
    writer = imageio.get_writer(out, fps=fps, macro_block_size=None)
    n = 0
    shape = None
    with Reader(bag) as r:
        conns = [c for c in r.connections if c.topic == args.topic]
        for con, _, raw in r.messages(connections=conns):
            msg = TS.deserialize_cdr(raw, con.msgtype)
            if con.msgtype.endswith("CompressedImage"):
                img = decode_compressed(msg, args.swap_rb)
            else:
                img = decode(msg)
            if img is None:
                writer.close()
                raise SystemExit("unsupported image encoding")
            writer.append_data(img)
            shape = img.shape
            n += 1
            if n % 100 == 0 or n == total:
                print(f"  {n}/{total} frames", flush=True)
            if args.max_frames and n >= args.max_frames:
                break
    writer.close()
    print(f"wrote {out}: {n} frames @ {fps:.1f} fps ({shape[1]}x{shape[0]})")


if __name__ == "__main__":
    main()
