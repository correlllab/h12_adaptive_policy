import threading

from .remote_controller import KeyMap


class KeyboardRemoteController:
    """Keyboard listener that mimics RemoteController.button states."""

    def __init__(self):
        self.button = [0] * 16
        self._lock = threading.Lock()
        self._listener = None

        try:
            from pynput import keyboard
        except ImportError as exc:
            raise ImportError(
                "Keyboard mode requires pynput. Install it with: pip install pynput"
            ) from exc

        self._keyboard = keyboard

        self._key_to_button = {
            "s": KeyMap.start,
            "a": KeyMap.A,
            "e": KeyMap.select,
            "q": KeyMap.L1,
            "w": KeyMap.R1,
            "x": KeyMap.X,
            "y": KeyMap.Y,
            "b": KeyMap.B,
            "u": KeyMap.up,
            "r": KeyMap.right,
            "d": KeyMap.down,
            "l": KeyMap.left,
        }

        self._listener = self._keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def _resolve_char(self, key):
        char = getattr(key, "char", None)
        if char is None:
            return None
        return char.lower()

    def _on_press(self, key):
        key_char = self._resolve_char(key)
        if key_char is None:
            return

        button_index = self._key_to_button.get(key_char)
        if button_index is None:
            return

        with self._lock:
            self.button[button_index] = 1

    def _on_release(self, key):
        key_char = self._resolve_char(key)
        if key_char is None:
            return

        button_index = self._key_to_button.get(key_char)
        if button_index is None:
            return

        with self._lock:
            self.button[button_index] = 0

    def set(self, _):
        """No-op to keep interface compatibility with RemoteController."""
        return

    def close(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def __del__(self):
        self.close()


def print_keyboard_mapping():
    print("Keyboard remote mapping:")
    print("  s -> START")
    print("  a -> A")
    print("  e -> SELECT (exit)")
    print("  q -> L1")
    print("  w -> R1")
    print("  b/x/y -> B/X/Y")
    print("  u/r/d/l -> up/right/down/left")
