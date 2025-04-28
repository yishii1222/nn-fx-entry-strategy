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
            # threaded=True がデフォルトなので非同期でポップアップを表示
            toaster.show_toast(title, message, duration=5)
        except TypeError:
            # WndProc が None を返す問題を無視
            pass
        except Exception as e:
            # その他のエラーは標準出力にフォールバック
            print(f"NOTIFICATION ERROR: {e}")
            print(f"NOTIFICATION: {title} - {message}")
    else:
        print(f"NOTIFICATION: {title} - {message}")
