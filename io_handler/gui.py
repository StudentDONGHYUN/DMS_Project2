try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("⚠️  tkinter를 사용할 수 없습니다. 파일 경로를 직접 입력해야 합니다.")

class DMS_GUI_Setup:
    pass

def get_user_input_terminal():
    pass
