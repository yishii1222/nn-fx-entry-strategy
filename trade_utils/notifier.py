try:
    import winsound
except ImportError:
    winsound = None

def beep():
    if winsound:
        try:
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            try:
                winsound.Beep(1000, 500)
            except Exception:
                pass
    else:
        import ctypes
        try:
            ctypes.windll.user32.MessageBeep(0xFFFFFFFF)
        except Exception:
            ctypes.windll.kernel32.Beep(1000, 500)

try:
    import win10toast
    from win10toast import ToastNotifier

    class PatchedToastNotifier(ToastNotifier):
        def on_destroy(self, hwnd, msg, wparam, lparam):
            try:
                super().on_destroy(hwnd, msg, wparam, lparam)
            except Exception:
                pass
            return 0

    toaster = PatchedToastNotifier()
except ImportError:
    toaster = None


def send_notification(title: str, message: str):
    if toaster:
        try:
            toaster.show_toast(title, message, duration=5, threaded=True)
        except Exception as e:
            print(f"NOTIFICATION ERROR: {e}")
            print(f"NOTIFICATION: {title} - {message}")
    else:
        print(f"NOTIFICATION: {title} - {message}")
