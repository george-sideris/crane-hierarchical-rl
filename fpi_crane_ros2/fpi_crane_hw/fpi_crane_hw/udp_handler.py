import socket
from typing import List, Tuple

from .robot_globals import RobotLinks, UdpMessage

# Data packet structure
# |----------------|-----------------|---------------|
# |Header (4 bytes)| Data (n bytes)  | CRC (2 bytes) |

MAX_DATA_SIZE = 992  # bytes
NUM_JOINTS = 9

DATA_START_INDEX = 4
NUM_JOINT_SENSORS = 5  # Mast, Boom, Stick, Telescope, Grapple base
GRAPPLE_DATA_START_INDEX = DATA_START_INDEX + NUM_JOINT_SENSORS * 4  # 4 bytes
PLC_STATE_START_INDEX = GRAPPLE_DATA_START_INDEX + 4  # 6 bytes
JOINT_EFFORTS_DATA_START_INDEX = PLC_STATE_START_INDEX + 6  # 10 bytes
RRC_VALUES_START_INDEX = 44  # 24 bytes: 12 uint16 values
PASSIVE_JOINT_VALUES_START_INDEX = RRC_VALUES_START_INDEX + 24  # 8 bytes


class UDPHandler:
    def __init__(
        self,
        recv_ip: str,
        recv_port: int,
        send_ip: str,
        send_port: int,
        recv_timeout: float = 0.005,
        # ACK wait per packet. 0.2 s was too tight: the PLC ACKs late right after finishing a
        # trajectory (comm task busy), and one late ACK rejects the whole next trajectory
        # ("Failed to send trajectory header"). Waiting longer is safe (no resend, no duplicates).
        send_timeout: float = 0.6,
    ) -> None:
        # UDP server socket for receiving data from PLC.
        self.recv_ip = recv_ip
        self.recv_port = int(recv_port)
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((self.recv_ip, self.recv_port))
        self.recv_sock.settimeout(recv_timeout)

        # UDP client socket for sending data to PLC and receiving ACKs.
        self.send_ip = send_ip
        self.send_port = int(send_port)
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_sock.settimeout(send_timeout)

    @staticmethod
    def compute_crc16(data: bytes) -> int:
        crc = 0x0000
        for value in data:
            crc ^= value << 8
            for _ in range(8):
                if (crc & 0x8000) > 0:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
        return crc & 0xFFFF

    @staticmethod
    def bytes_to_int(value: bytes) -> int:
        return int.from_bytes(value, byteorder="big", signed=True)

    @staticmethod
    def bytes_to_uint(value: bytes) -> int:
        return int.from_bytes(value, byteorder="big", signed=False)

    def pack_data(self, packet_id: int, data: bytes) -> List[bytes]:
        packets = []
        data_size = len(data)
        total_packets = (data_size // MAX_DATA_SIZE) + (1 if data_size % MAX_DATA_SIZE else 0)

        for i in range(total_packets):
            start = i * MAX_DATA_SIZE
            end = min((i + 1) * MAX_DATA_SIZE, data_size)
            chunk = data[start:end]

            packet_id_bytes = int(packet_id).to_bytes(2, "big", signed=False)
            packet_length_bytes = len(chunk).to_bytes(2, "big", signed=False)
            crc_value = self.compute_crc16(packet_id_bytes + packet_length_bytes + chunk)
            crc_bytes = crc_value.to_bytes(2, "big", signed=False)
            packets.append(packet_id_bytes + packet_length_bytes + chunk + crc_bytes)

        return packets

    def send_message(self, msg_id: int, data: bytes) -> bool:
        packets = self.pack_data(msg_id, data)
        sent_count = 0
        for packet in packets:
            if self.send_packet_and_receive_ack(packet):
                sent_count += 1
        return sent_count == len(packets)

    def send_packet_and_receive_ack(self, packet: bytes) -> bool:
        self.send_sock.sendto(packet, (self.send_ip, self.send_port))
        try:
            data, _ = self.send_sock.recvfrom(1024)
        except socket.timeout:
            return False

        if len(data) < 2:
            return False
        ack_id = self.bytes_to_uint(data[0:2])
        return ack_id == int(UdpMessage.MSG_ACK)

    def receive_message(self, buffer_size: int = 1024) -> Tuple[int, bytes, tuple]:
        data, addr = self.recv_sock.recvfrom(buffer_size)
        if len(data) < DATA_START_INDEX + 2:
            raise ValueError(f"UDP packet too short: {len(data)} bytes")

        message_id = self.bytes_to_uint(data[0:2])
        message_len = self.bytes_to_uint(data[2:4])
        crc_start = DATA_START_INDEX + message_len
        crc_end = crc_start + 2
        if len(data) < crc_end:
            raise ValueError(
                f"UDP packet length mismatch: header says {message_len} bytes, "
                f"but packet has {len(data)} bytes"
            )

        received_crc = self.bytes_to_uint(data[crc_start:crc_end])
        computed_crc = self.compute_crc16(data[:crc_start])
        if received_crc != computed_crc:
            raise ValueError("CRC mismatch: received CRC does not match computed CRC")

        return message_id, data, addr

    def unpack_joint_states(self, msg_data: bytes):
        sensor_data = msg_data[DATA_START_INDEX : DATA_START_INDEX + 20]
        effort_data = msg_data[JOINT_EFFORTS_DATA_START_INDEX : JOINT_EFFORTS_DATA_START_INDEX + 10]

        positions = [0] * NUM_JOINTS
        velocities = [0] * NUM_JOINTS
        efforts = [0] * NUM_JOINTS

        joint_sensors_order = [
            RobotLinks.MAST,
            RobotLinks.BOOM,
            RobotLinks.STICK,
            RobotLinks.TELESCOPE,
            RobotLinks.GRAPPLE_BASE,
        ]

        for i, joint_index in enumerate(joint_sensors_order):
            positions[joint_index] = self.bytes_to_int(sensor_data[i * 4 : i * 4 + 2])
            velocities[joint_index] = self.bytes_to_int(sensor_data[i * 4 + 2 : i * 4 + 4])
            efforts[joint_index] = self.bytes_to_int(effort_data[i * 2 : i * 2 + 2])

        return positions, velocities, efforts

    def unpack_passive_joint_states(self, msg_data: bytes):
        joint_data = msg_data[PASSIVE_JOINT_VALUES_START_INDEX : PASSIVE_JOINT_VALUES_START_INDEX + 8]
        positions = [0] * 2
        velocities = [0] * 2

        for i in range(2):
            positions[i] = self.bytes_to_int(joint_data[i * 4 : i * 4 + 2])
            velocities[i] = self.bytes_to_int(joint_data[i * 4 + 2 : i * 4 + 4])

        return positions, velocities

    def unpack_grapple_states(self, msg_data: bytes):
        grapple_data = msg_data[GRAPPLE_DATA_START_INDEX : GRAPPLE_DATA_START_INDEX + 4]

        move_completeness = self.bytes_to_uint(grapple_data[0:1])
        move_angle = self.bytes_to_uint(grapple_data[1:3])
        grapple_state = self.bytes_to_uint(grapple_data[3:4])

        return move_completeness, move_angle, grapple_state

    def unpack_plc_states(self, msg_data: bytes):
        plc_data = msg_data[PLC_STATE_START_INDEX : PLC_STATE_START_INDEX + 6]

        plc_state = self.bytes_to_uint(plc_data[0:1])
        plc_move_sequence_id = self.bytes_to_uint(plc_data[2:4])
        plc_move_point_id = self.bytes_to_uint(plc_data[4:6])

        return plc_state, plc_move_sequence_id, plc_move_point_id

    def unpack_rrc_values(self, msg_data: bytes):
        rrc_data = msg_data[RRC_VALUES_START_INDEX : RRC_VALUES_START_INDEX + 24]
        return [self.bytes_to_uint(rrc_data[i * 2 : i * 2 + 2]) for i in range(12)]

    def close(self) -> None:
        self.send_sock.close()
        self.recv_sock.close()
