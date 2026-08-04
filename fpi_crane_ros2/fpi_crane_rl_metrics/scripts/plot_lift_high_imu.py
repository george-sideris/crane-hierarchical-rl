#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from build.lib.fpi_crane_rl_metrics.stability_service_node import quaternion_to_rotation_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("imu_csv")
    parser.add_argument("--window-s", type=float, default=1.0)
    args = parser.parse_args()

    df = pd.read_csv(args.imu_csv)

    # Drop empty IMU messages that are encoded as all-zero measurements.
    zero_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
    if all(col in df.columns for col in zero_cols):
        rows_before = len(df)
        df = df.loc[~(df[zero_cols] == 0).all(axis=1)]
        rows_after = len(df)
        print(f"Dropped {rows_before - rows_after} all-zero IMU rows")

    t = df["t_offset_s"].to_numpy()
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    window = max(3, int(args.window_s * fs))

    ax = df["ax"].to_numpy()
    ay = df["ay"].to_numpy()
    az = df["az"].to_numpy()

    gx = df["gx"].to_numpy()
    gy = df["gy"].to_numpy()
    gz = df["gz"].to_numpy()

    qx = df["qx"].to_numpy()
    qy = df["qy"].to_numpy()
    qz = df["qz"].to_numpy()
    qw = df["qw"].to_numpy()

    a_norm = np.sqrt(ax**2 + ay**2 + az**2)
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2)

    # Remove slow acceleration/gravity trend using a rolling mean.
    # This is a simple first-pass vibration estimate, not final filtering.
    a_df = pd.DataFrame({"ax": ax, "ay": ay, "az": az})
    a_slow = a_df.rolling(window=window, center=True, min_periods=1).mean()
    a_hp = a_df - a_slow
    a_hp_norm = np.sqrt(
        a_hp["ax"].to_numpy()**2 +
        a_hp["ay"].to_numpy()**2 +
        a_hp["az"].to_numpy()**2
    )

    quiet_score = (
        pd.Series(a_hp_norm).rolling(window=window, center=True, min_periods=1).mean()
        + pd.Series(g_norm).rolling(window=window, center=True, min_periods=1).mean()
    )

    quiet_idx = int(np.argmin(quiet_score))
    quiet_t = t[quiet_idx]

    # Mean rotation matrix from quaternion, used to estimate gravity direction.
    q_mean = np.mean(np.stack([qx, qy, qz, qw], axis=1), axis=0)
    x, y, z, w = q_mean
    R_earth_imu_mean = quaternion_to_rotation_matrix(x, y, z, w)

    print(f"Estimated sample rate: {fs:.2f} Hz")
    print(f"Quietest approximate time: bag offset {quiet_t:.3f} s")
    print(f"Acceleration norm at quiet point: {a_norm[quiet_idx]:.3f} m/s^2")
    print(f"Gyro norm at quiet point: {g_norm[quiet_idx]:.5f} rad/s")
    print(f"Vibration estimate at quiet point: {a_hp_norm[quiet_idx]:.5f} m/s^2")
    print(f"Mean rotation matrix from quaternions:\n{R_earth_imu_mean}")
    print(f"Estimated gravity direction in IMU frame: {R_earth_imu_mean.T[:, 2]}")

    plt.figure()
    plt.plot(t, ax, label="ax")
    plt.plot(t, ay, label="ay")
    plt.plot(t, az, label="az")
    plt.xlabel("Bag offset time [s]")
    plt.ylabel("Linear acceleration [m/s²]")
    plt.legend()
    plt.grid(True)
    plt.title("ZED IMU linear acceleration during LIFT_HIGH")
    plt.savefig("imu_accel_lift_high.png", dpi=200)

    plt.figure()
    plt.plot(t, qx, label="qx")
    plt.plot(t, qy, label="qy")
    plt.plot(t, qz, label="qz")
    plt.plot(t, qw, label="qw")
    plt.xlabel("Bag offset time [s]")
    plt.ylabel("Quaternion components")
    plt.legend()
    plt.grid(True)
    plt.title("ZED IMU quaternion during LIFT_HIGH")
    plt.savefig("imu_quaternion_lift_high.png", dpi=200)

    plt.figure()
    plt.plot(t, a_norm)
    plt.axvline(quiet_t, linestyle="--")
    plt.xlabel("Bag offset time [s]")
    plt.ylabel("||a|| [m/s²]")
    plt.grid(True)
    plt.title("Acceleration norm during LIFT_HIGH")
    plt.savefig("imu_accel_norm_lift_high.png", dpi=200)

    plt.figure()
    plt.plot(t, g_norm)
    plt.axvline(quiet_t, linestyle="--")
    plt.xlabel("Bag offset time [s]")
    plt.ylabel("||gyro|| [rad/s]")
    plt.grid(True)
    plt.title("Gyro norm during LIFT_HIGH")
    plt.savefig("imu_gyro_norm_lift_high.png", dpi=200)

    plt.figure()
    plt.plot(t, a_hp_norm)
    plt.axvline(quiet_t, linestyle="--")
    plt.xlabel("Bag offset time [s]")
    plt.ylabel("High-pass acceleration norm [m/s²]")
    plt.grid(True)
    plt.title("Simple vibration estimate during LIFT_HIGH")
    plt.savefig("imu_vibration_estimate_lift_high.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()