#!/usr/bin/env python3
import socket

import rclpy
from rclpy.node import Node

from .robot_globals import UdpMessage
from .udp_handler import UDPHandler


class PLCEmulator(Node):
    def __init__(self) -> None:
        super().__init__("plc_emulator")
        self.declare_parameter("src_ip", "172.20.230.120")
        self.declare_parameter("src_port", 30310)
        self.declare_parameter("remote_ip", "172.20.230.162")
        self.declare_parameter("remote_port", 30305)
        self.declare_parameter("udp_recv_timeout", 0.005)
        self.declare_parameter("poll_rate", 100.0)

        self.src_ip = str(self.get_parameter("src_ip").value)
        self.src_port = int(self.get_parameter("src_port").value)
        self.remote_ip = str(self.get_parameter("remote_ip").value)
        self.remote_port = int(self.get_parameter("remote_port").value)
        timeout = float(self.get_parameter("udp_recv_timeout").value)

        self.udp = UDPHandler(self.src_ip, self.src_port, self.remote_ip, self.remote_port, timeout)
        self.timer = self.create_timer(1.0 / float(self.get_parameter("poll_rate").value), self.poll_once)
        self.get_logger().info(
            f"PLC emulator running. Listening on {self.src_ip}:{self.src_port}; "
            f"ACKs sent to {self.remote_ip}:{self.remote_port}"
        )

    def poll_once(self) -> None:
        try:
            msg_id, _, _ = self.udp.receive_message()
        except socket.timeout:
            return
        except ValueError as exc:
            self.get_logger().warn(f"Invalid packet received: {exc}")
            return

        self.get_logger().info(f"Received message ID: {msg_id}")
        ack_payload = b"ACK"
        ack_packet = self.udp.pack_data(int(UdpMessage.MSG_ACK), ack_payload)[0]
        self.udp.send_sock.sendto(ack_packet, (self.remote_ip, self.remote_port))

    def destroy_node(self) -> bool:
        self.udp.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PLCEmulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
