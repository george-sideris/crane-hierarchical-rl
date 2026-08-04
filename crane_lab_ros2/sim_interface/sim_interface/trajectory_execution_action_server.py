#!/usr/bin/env python3

import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from sim_interface.action import FollowPath
from sim_interface.srv import SetTarget, CheckTargetReached
from sim_interface.path_planner import PathPlanner


WAIT_FOR_TARGET_REACHED = True
TARGET_REACH_TOLERANCE = 0.05

class TrajectoryExecutionActionServer(Node):
    def __init__(self, wait_for_target_reached=WAIT_FOR_TARGET_REACHED, target_reach_tolerance=TARGET_REACH_TOLERANCE):
        super().__init__('trajectory_execution_action_server')
        self.planner = PathPlanner()
        self.wait_for_target_reached = wait_for_target_reached
        self.target_reach_tolerance = target_reach_tolerance
        self.set_target_client = self.create_client(SetTarget, '/set_target')
        self.reach_check_client = self.create_client(CheckTargetReached, '/is_target_reached')
        while not self.set_target_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /set_target...")
        while not self.reach_check_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /is_target_reached...")
        self.action_server = ActionServer(self, FollowPath, 'follow_path', self.execute_callback)
        self.get_logger().info("TrajectoryExecutionActionServer is ready.")

    async def execute_callback(self, goal_handle):
        self.get_logger().info("Received FollowPath goal.")
        goal = goal_handle.request
        path_array = []
        for wp in goal.waypoints:
            x, y, z = wp.position
            yaw = wp.yaw
            opening = wp.grapple_opening
            path_array.append([x, y, z, yaw, opening])
        waypoints_array = np.array(path_array)
        try:
            path = self.generate_path(goal.path_type, waypoints_array, goal.resolution)
        except Exception as e:
            self.get_logger().error(f"Path generation failed: {e}")
            goal_handle.abort()
            return FollowPath.Result(success=False)
        total = len(path)
        for i, pose in enumerate(path):
            if goal_handle.is_cancel_requested:
                self.get_logger().warn("Goal canceled.")
                goal_handle.canceled()
                return FollowPath.Result(success=False)
            req = SetTarget.Request()
            req.target_xyz = pose[:3].tolist()
            req.target_yaw = float(pose[3])
            req.grapple_opening = float(pose[4])
            future = self.set_target_client.call_async(req)
            await future
            if future.result() is None:
                self.get_logger().error("SetTarget service failed.")
                goal_handle.abort()
                return FollowPath.Result(success=False)
            if self.wait_for_target_reached:
                target_reached = await self.wait_until_target_reached(tolerance = self.target_reach_tolerance, timeout=2.0)
                if not target_reached:
                    self.get_logger().warn("Target not reached in time, continuing.")
            feedback = FollowPath.Feedback()
            feedback.percent_complete = (i + 1) / total * 100.0
            goal_handle.publish_feedback(feedback)
        self.get_logger().info("Path execution completed.")
        goal_handle.succeed()
        return FollowPath.Result(success=True)

    async def wait_until_target_reached(self, tolerance: float = None, timeout: float = 2.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            req = CheckTargetReached.Request()
            if tolerance:
                req.tolerance = tolerance
            future = self.reach_check_client.call_async(req)
            await future
            if future.result() and future.result().target_reached:
                return True
        return False

    def generate_path(self, path_type: str, waypoints_array: np.ndarray, resolution: float = None):
        if path_type == "linear":
            return self.planner.plan_linear_path(waypoints_array, resolution)
        elif path_type == "cubic_spline":
            return self.planner.plan_cubic_spline_path(waypoints_array, resolution)
        else:
            raise ValueError(f"Unsupported path_type: {path_type}")


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryExecutionActionServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
