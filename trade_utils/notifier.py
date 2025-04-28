try:
    import winsound
except ImportError:
    winsound = None

def beep():
    if winsound:
        winsound.Beep(1000, 500)
    else:
        import ctypes
        ctypes.windll.kernel32.Beep(1000, 500)

try:
    import win10toast
    from win10toast import ToastNotifier

    class PatchedToastNotifier(ToastNotifier):
        def on_destroy(self, hwnd, msg, wparam, lparam):
            # base の通知終了処理を行いつつ、整数を返す
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
            # 非同期実行のまま、patched on_destroy でエラー回避
            toaster.show_toast(title, message, duration=5, threaded=True)
        except Exception as e:
            # 何らかの例外は標準出力にフォールバック
            print(f"NOTIFICATION ERROR: {e}")
            print(f"NOTIFICATION: {title} - {message}")
    else:
        print(f"NOTIFICATION: {title} - {message}")
