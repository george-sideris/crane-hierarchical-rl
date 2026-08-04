# Clock sync between the perception kit and the laptop

The ZED perception kit box (172.20.230.118) stamps camera frames with its own
clock. That clock has no time source, so it drifts and lands at a different
offset every boot (measured: +6.5 s on 2026-07-03, -1.1 s on 2026-07-20,
+3.5 s live on 2026-07-27). Everything published on the laptop (tf,
joint_states, plc_status) uses the laptop clock, so ZED data appears seconds
out of sync when aligned by header stamp. This also trips the stamp-based
cloud freshness check in crane_policy_node.

## Fix at the source: chrony

The laptop serves NTP on the crane LAN; the kit syncs to it.

Laptop (172.20.230.162):

    sudo apt install chrony            # if not installed
    sudo sh -c 'cat chrony_laptop.conf >> /etc/chrony/chrony.conf'
    sudo systemctl restart chrony

Kit box (172.20.230.118):

    sudo apt install chrony            # needs internet once; for an offline
                                       # Jetson, download the arm64 .deb on a
                                       # machine with internet and scp it over
    # edit /etc/chrony/chrony.conf: replace the pool/server lines with the
    # contents of chrony_kit.conf
    sudo systemctl restart chrony
    chronyc tracking                   # Reference ID should be 172.20.230.162
                                       # and System time offset < 1 ms

Verify end to end on the laptop with the wrapper running:

    ros2 topic delay /zed_0/zed_node/point_cloud/cloud_registered
    # should read ~0.06 s (real transport latency), not seconds

## Fix already-recorded bags: restamp_zed_bag.py

Estimates the per-bag offset (median of receive time minus header stamp over
the ZED topics; constant within a session, ~10 ms jitter) and shifts the ZED
header stamps by it. Receive timestamps and all non-ZED topics are untouched.
Works directly on rosbag2 sqlite3 bags, no ROS environment needed.

    python3 restamp_zed_bag.py <bag_dir> --dry-run   # report offsets only
    python3 restamp_zed_bag.py <bag_dir>             # writes <bag_dir>_restamped

Options: --out DIR, --topic-filter STR (default /zed_node/), --latency SEC
(true transport latency to keep in the stamps, ~0.06), --offset SEC (override
the estimate). The original bag is never modified.

Caveat: restamping aligns the ZED clock to the laptop clock but cannot
recover sub-offset accuracy better than the transport latency; for
calibration-grade timing, sync the clocks before recording.
