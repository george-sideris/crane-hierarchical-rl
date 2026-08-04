from __future__ import annotations

import atexit
import select
import signal
import sys
import threading
from typing import Iterable, List, Set, Tuple

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Joy
except ImportError:  # pragma: no cover - allows unit tests without ROS installed
    rclpy = None
    Node = object
    Joy = None


def build_joy_state(pressed_keys: Iterable[str], axes: List[float] | None = None, buttons: List[int] | None = None) -> Tuple[List[float], List[int]]:
    axes_state = [0.0] * 6 if axes is None else list(axes)
    button_state = [0] * 6 if buttons is None else list(buttons)
    pressed = set(pressed_keys)

    if "a" in pressed:
        axes_state[0] = -1.0
    if "d" in pressed:
        axes_state[0] = 1.0
    if "w" in pressed:
        axes_state[1] = -1.0
    if "s" in pressed:
        axes_state[1] = 1.0

    if "j" in pressed:
        axes_state[3] = -1.0
    if "l" in pressed:
        axes_state[3] = 1.0
    if "i" in pressed:
        axes_state[4] = -1.0
    if "k" in pressed:
        axes_state[4] = 1.0

    if "q" in pressed:
        button_state[4] = 1
    if "e" in pressed:
        button_state[5] = 1

    if " " in pressed:
        axes_state = [0.0] * 6
        button_state = [0] * 6

    return axes_state, button_state


class KeyboardJoyNode(Node):
    def __init__(self) -> None:
        super().__init__("keyboard_joy_node")
        self._publisher = self.create_publisher(Joy, "/joy", 10)
        self._timer = self.create_timer(0.05, self._publish_state)
        self._lock = threading.Lock()
        self._pending_keys: Set[str] = set()
        self._sticky_modifiers: Set[str] = set()
        self._terminal_state = None
        self._input_thread = threading.Thread(target=self._read_input, daemon=True)
        self._input_thread.start()

        atexit.register(self._restore_terminal)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.get_logger().info(
            "Keyboard joystick emulation started. Use: w/s (left stick), a/d (left horizontal), "
            "i/k (right vertical), j/l (right horizontal), q/e (mode toggle), and space to clear."
        )

    def _publish_state(self) -> None:
        if Joy is None:
            return

        with self._lock:
            pending = set(self._pending_keys)
            self._pending_keys.clear()
            modifiers = set(self._sticky_modifiers)

        axes, buttons = build_joy_state(pending | modifiers)
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = axes
        msg.buttons = buttons
        self._publisher.publish(msg)

    def _read_input(self) -> None:
        if not sys.stdin.isatty():
            self.get_logger().warn("stdin is not a TTY; keyboard input is unavailable.")
            return

        try:
            import termios
            import tty
        except ImportError:  # pragma: no cover
            return

        fd = sys.stdin.fileno()
        original_attrs = termios.tcgetattr(fd)
        self._terminal_state = (fd, original_attrs)
        try:
            tty.setcbreak(fd)
            while rclpy is not None and rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    if not char:
                        continue
                    if char in {"\x03", "\x1b"}:
                        break
                    self._handle_key(char)
        finally:
            self._restore_terminal()

    def _handle_key(self, char: str) -> None:
        char = char.lower()
        if char == " ":
            with self._lock:
                self._pending_keys.clear()
                self._sticky_modifiers.clear()
            return

        if char in {"q", "e"}:
            with self._lock:
                if char in self._sticky_modifiers:
                    self._sticky_modifiers.remove(char)
                else:
                    self._sticky_modifiers.add(char)
            return

        if char in {"a", "d", "w", "s", "j", "l", "i", "k"}:
            with self._lock:
                self._pending_keys.add(char)

    def _restore_terminal(self) -> None:
        if self._terminal_state is None:
            return

        fd, original_attrs = self._terminal_state
        try:
            import termios
            termios.tcsetattr(fd, termios.TCSADRAIN, original_attrs)
        except Exception:
            pass
        finally:
            self._terminal_state = None

    def _signal_handler(self, signum, frame) -> None:
        self._restore_terminal()
        if signum in {signal.SIGINT, signal.SIGTERM}:
            raise SystemExit(0)


def main(args=None) -> None:
    if rclpy is None:
        raise RuntimeError("rclpy is required to run the keyboard joystick node")

    rclpy.init(args=args)
    node = KeyboardJoyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
