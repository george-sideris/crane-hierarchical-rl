#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import tkinter as tk
from tkinter import ttk
import math
from threading import Thread

class RobotControlUI(Node):
    def __init__(self, root):
        super().__init__('robot_velocity_ui')
        self.root = root
        self.root.title("Robot Velocity Controller")
        self.joint_names = ["slew_joint", "boom_joint", "cabin_swing_up_joint", "stick_joint", "cabin_swing_down_joint", 
            "hanger_joint", "bearingfork_joint", "grapplecarrier_joint", "grappletong1_joint", "grappletong2_joint"]
        self.sliders = {}
        self.values = {}
        self.positions = {}
        self.velocities = {}

        for i, joint in enumerate(self.joint_names):
            ttk.Label(root, text=f"{joint}").grid(row=i, column=0, padx=5, pady=5)
            value_label = ttk.Label(root, text="0.0")
            value_label.grid(row=i, column=2, padx=5, pady=5)
            self.values[joint] = value_label
            slider = ttk.Scale(root, from_=-0.5, to=0.5, length=200, orient='horizontal',
                               command=lambda v, j=joint: self.update_value(j, v))
            slider.set(0.0)
            slider.grid(row=i, column=1, padx=5, pady=5)
            self.sliders[joint] = slider
            zero_button = ttk.Button(root, text="Zero", command=lambda j=joint: self.zero_slider(j))
            zero_button.grid(row=i, column=3, padx=5, pady=5)
            position_label = ttk.Label(root, text="Pos: 0.0")
            position_label.grid(row=i, column=4, padx=5, pady=5)
            self.positions[joint] = position_label
            velocity_label = ttk.Label(root, text="Vel: 0.0")
            velocity_label.grid(row=i, column=5, padx=5, pady=5)
            self.velocities[joint] = velocity_label

        self.zero_all_button = ttk.Button(root, text="Zero All", command=self.zero_all_sliders)
        self.zero_all_button.grid(row=len(self.joint_names), columnspan=6, pady=10)

        self.pub = self.create_publisher(JointState, '/joint_command', 10)
        self.sub = self.create_subscription(JointState, '/joint_states_sim', self.joint_state_callback, 10)

        self.timer = self.create_timer(0.015, self.update_commands)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_value(self, joint, value):
        self.values[joint].config(text=f"{math.degrees(float(value)):.2f}")

    def update_commands(self):
        msg = JointState()
        now = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.name = self.joint_names
        msg.velocity = [self.sliders[joint].get() for joint in self.joint_names]
        self.pub.publish(msg)

    def joint_state_callback(self, msg):
        joint_data = dict(zip(msg.name, zip(msg.position, msg.velocity)))
        for joint in self.joint_names:
            if joint in joint_data:
                pos, vel = joint_data[joint]
                self.positions[joint].config(text=f"Pos: {math.degrees(pos):.2f}")
                self.velocities[joint].config(text=f"Vel: {math.degrees(vel):.2f}")

    def zero_slider(self, joint):
        self.sliders[joint].set(0.0)
        self.update_value(joint, 0.0)

    def zero_all_sliders(self):
        for joint in self.joint_names:
            self.zero_slider(joint)

    def on_close(self):
        self.destroy_node()
        rclpy.shutdown()
        self.root.destroy()


def run_ros2_node(root):
    rclpy.init()
    node = RobotControlUI(root)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


def main():
    root = tk.Tk()
    root.title("Robot Velocity Controller")
    ros_thread = Thread(target=run_ros2_node, args=(root,), daemon=True)
    ros_thread.start()
    root.mainloop()

if __name__ == '__main__':
    main()
