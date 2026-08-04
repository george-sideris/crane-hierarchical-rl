from enum import IntEnum


class RobotLinks(IntEnum):
    MAST = 0
    BOOM = 1
    STICK = 2
    TELESCOPE = 3
    UPPERPASSIVE = 4
    LOWERPASSIVE = 5
    GRAPPLE_BASE = 6
    GRIPPER_LEFT = 7
    GRIPPER_RIGHT = 8


class GripperDirection(IntEnum):
    NONE = 0
    OPEN = 1
    CLOSE = 2


class PlcState(IntEnum):
    INIT = 0
    HOLD = 1
    EXECUTE_SEQUENCE = 2
    END_SEQUENCE = 3


class UdpMessage(IntEnum):
    MSG_TRAJ_HEADER = 1
    MSG_TRAJ = 2
    MSG_START_STOP_EXECUTION = 3
    MSG_CURENT_PLC_VALUES = 4  # Kept spelling to match the PLC protocol.
    MSG_SET_GRAPPLE_POS = 5
    MSG_ACK = 6
    MSG_NACK = 7

class RRCAxes(IntEnum):
    MAST = 0
    BOOM = 1
    TELESCOPE = 2
    GRAPPLE_ROTATION = 3
    STICK = 4
    GRAPPLE_OPEN= 5