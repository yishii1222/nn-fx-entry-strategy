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
    from win10toast import ToastNotifier
    toaster = ToastNotifier()
except ImportError:
    toaster = None

def send_notification(title: str, message: str):
    if toaster:
        try:
            toaster.show_toast(title, message, duration=5)
        except Exception as e:
            # 通知エラー時は標準出力にフォールバック
            print(f"NOTIFICATION ERROR: {e}")
            print(f"NOTIFICATION: {title} - {message}")
    else:
        print(f"NOTIFICATION: {title} - {message}")
