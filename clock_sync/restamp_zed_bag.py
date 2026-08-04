#!/usr/bin/env python3
"""Fix ZED header stamps in rosbag2 (sqlite3) bags recorded with an unsynced camera clock.

The ZED perception kit box stamps frames with its own clock, which is not NTP
synced and can be several seconds off from the laptop that records the bag.
All other topics (tf, joint_states, plc_status, ...) are stamped with the
laptop clock, so ZED data appears seconds out of sync when aligning by header
stamp.

The offset is constant within a recording session (jitter ~10 ms), so it can
be estimated per bag as the median of (bag receive time - header stamp) over
the ZED messages, and removed by shifting the header stamps. Bag receive
times (the sqlite 'timestamp' column) are laptop clock and are left untouched.

Usage:
    python3 restamp_zed_bag.py <bag_dir>                # writes <bag_dir>_restamped
    python3 restamp_zed_bag.py <bag_dir> --dry-run      # only report offsets
    python3 restamp_zed_bag.py <bag_dir> --out <dir>
    python3 restamp_zed_bag.py <bag_dir> --latency 0.06 # subtract known transport latency

By default only topics whose name contains '/zed_node/' are touched.
"""

import argparse
import glob
import os
import shutil
import sqlite3
import statistics
import struct
import sys

# Message types whose serialized CDR layout starts with std_msgs/Header,
# i.e. stamp sec (int32) at byte 4 and nanosec (uint32) at byte 8, right
# after the 4-byte CDR encapsulation header.
HEADER_FIRST_TYPES = {
    'sensor_msgs/msg/PointCloud2',
    'sensor_msgs/msg/Image',
    'sensor_msgs/msg/CompressedImage',
    'sensor_msgs/msg/CameraInfo',
    'sensor_msgs/msg/Imu',
    'sensor_msgs/msg/JointState',
    'sensor_msgs/msg/MagneticField',
    'sensor_msgs/msg/Temperature',
    'sensor_msgs/msg/FluidPressure',
    'nav_msgs/msg/Odometry',
    'geometry_msgs/msg/PoseStamped',
    'geometry_msgs/msg/PoseWithCovarianceStamped',
    'geometry_msgs/msg/TwistStamped',
    'geometry_msgs/msg/TransformStamped',
}


def read_stamp_ns(data):
    """Return the header stamp in nanoseconds, or None if not parseable."""
    if len(data) < 12:
        return None
    # CDR encapsulation: 0x00 0x01 = little endian, 0x00 0x00 = big endian.
    if data[0] != 0x00 or data[1] != 0x01:
        return None
    sec, nanosec = struct.unpack_from('<iI', data, 4)
    return sec * 1_000_000_000 + nanosec


def write_stamp_ns(data, stamp_ns):
    sec, nanosec = divmod(stamp_ns, 1_000_000_000)
    return data[:4] + struct.pack('<iI', sec, nanosec) + data[12:]


def collect_target_topics(db, topic_filter):
    """Return ({topic_id: name}, [skipped names]) for topics matching the filter."""
    targets = {}
    skipped = []
    for topic_id, name, msg_type in db.execute('SELECT id, name, type FROM topics'):
        if topic_filter not in name:
            continue
        if msg_type in HEADER_FIRST_TYPES:
            targets[topic_id] = name
        else:
            skipped.append('%s (%s)' % (name, msg_type))
    return targets, skipped


def topic_deltas(db, targets):
    """Return {topic_name: [receive_ns - stamp_ns, ...]} for target topics."""
    deltas = {name: [] for name in targets.values()}
    for topic_id, name in targets.items():
        rows = db.execute(
            'SELECT timestamp, data FROM messages WHERE topic_id = ?', (topic_id,))
        for receive_ns, data in rows:
            stamp_ns = read_stamp_ns(data)
            if stamp_ns is None or stamp_ns == 0:
                continue
            deltas[name].append(receive_ns - stamp_ns)
    return deltas


def apply_offset(db, targets, offset_ns):
    """Shift header stamps of target topics by offset_ns. Returns count."""
    updated = 0
    cur = db.cursor()
    for topic_id in targets:
        rows = cur.execute(
            'SELECT id, data FROM messages WHERE topic_id = ?', (topic_id,)).fetchall()
        for msg_id, data in rows:
            stamp_ns = read_stamp_ns(data)
            if stamp_ns is None or stamp_ns == 0:
                continue
            new_data = write_stamp_ns(data, stamp_ns + offset_ns)
            cur.execute('UPDATE messages SET data = ? WHERE id = ?',
                        (sqlite3.Binary(new_data), msg_id))
            updated += 1
    db.commit()
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('bag_dir', help='rosbag2 directory (contains *.db3 and metadata.yaml)')
    parser.add_argument('--out', default=None,
                        help='output bag directory (default: <bag_dir>_restamped)')
    parser.add_argument('--topic-filter', default='/zed_node/',
                        help='only topics containing this substring are restamped')
    parser.add_argument('--latency', type=float, default=0.0,
                        help='known transport latency in seconds to keep in the stamps '
                             '(subtracted from the estimated offset)')
    parser.add_argument('--offset', type=float, default=None,
                        help='use this offset in seconds instead of estimating it')
    parser.add_argument('--dry-run', action='store_true',
                        help='estimate and report offsets without writing anything')
    args = parser.parse_args()

    bag_dir = args.bag_dir.rstrip('/')
    if not os.path.isdir(bag_dir):
        sys.exit('not a directory: %s' % bag_dir)
    db_files = sorted(glob.glob(os.path.join(bag_dir, '*.db3')))
    if not db_files:
        sys.exit('no .db3 files in %s' % bag_dir)

    # Estimate the offset across all db3 files of the bag (one clock, one session).
    all_deltas = []
    per_topic = {}
    skipped_all = set()
    for db_file in db_files:
        db = sqlite3.connect(db_file)
        targets, skipped = collect_target_topics(db, args.topic_filter)
        skipped_all.update(skipped)
        for name, deltas in topic_deltas(db, targets).items():
            per_topic.setdefault(name, []).extend(deltas)
            all_deltas.extend(deltas)
        db.close()

    if not all_deltas:
        sys.exit('no restampable messages matched filter %r' % args.topic_filter)

    print('bag: %s' % bag_dir)
    for name in sorted(per_topic):
        if per_topic[name]:
            print('  %-55s median offset %+.3f s  (%d msgs)'
                  % (name, statistics.median(per_topic[name]) / 1e9, len(per_topic[name])))
    for name in sorted(skipped_all):
        print('  skipped (no leading header): %s' % name)

    estimated_ns = int(statistics.median(all_deltas))
    medians = [statistics.median(d) for d in per_topic.values() if d]
    spread = (max(medians) - min(medians)) / 1e9 if len(medians) > 1 else 0.0
    if spread > 0.2:
        print('warning: per-topic offsets differ by %.3f s; '
              'a single shared offset may not fit all topics' % spread)

    if args.offset is not None:
        offset_ns = int(args.offset * 1e9)
        print('using explicit offset %+.3f s' % args.offset)
    else:
        offset_ns = estimated_ns - int(args.latency * 1e9)
        print('estimated offset %+.3f s, applying %+.3f s (latency %.3f s kept)'
              % (estimated_ns / 1e9, offset_ns / 1e9, args.latency))

    if args.dry_run:
        print('dry run, nothing written')
        return

    out_dir = args.out or bag_dir + '_restamped'
    if os.path.exists(out_dir):
        sys.exit('output already exists: %s' % out_dir)
    print('copying to %s ...' % out_dir)
    shutil.copytree(bag_dir, out_dir)

    total = 0
    for db_file in sorted(glob.glob(os.path.join(out_dir, '*.db3'))):
        db = sqlite3.connect(db_file)
        targets, _ = collect_target_topics(db, args.topic_filter)
        total += apply_offset(db, targets, offset_ns)

        # Verify: residual offset after the fix should be ~0 (+ transport latency).
        residuals = [d for deltas in topic_deltas(db, targets).values() for d in deltas]
        db.close()
        if residuals:
            print('  %s: residual median %+.3f s'
                  % (os.path.basename(db_file), statistics.median(residuals) / 1e9))
    print('restamped %d messages -> %s' % (total, out_dir))


if __name__ == '__main__':
    main()
