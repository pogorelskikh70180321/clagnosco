# Информация о ВКР "Clagnosco":
#  ФИО автора: Погорельских Константин Владимирович
#  Тема ВКР: «Классификация изображений с помощью искусственного интеллекта (на примере Частного образовательного учреждения высшего образования «Московский университет имени С.Ю. Витте»).»
#  ВУЗ: ЧОУ ВО «Московский университет им. С.Ю. Витте»
#  Специальность: Прикладная информатика [09.03.03] Бакалавр
#  Факультет: Информационных технологий
#  Специализация / Профиль подготовки: Искусственный интеллект и анализ данных
#  Учебная группа: ИД 23.3/Б3-21

print("Clagnosco Start")

import threading
import time
import webbrowser
import os
import tkinter as tk
import sys

try:
    import google.colab
    IS_COLAB = True
except:
    IS_COLAB = False


PROJECT_VERSION = "1.0.0"

HOST_LINK = '127.0.0.1'
PORT_LINK = 5000

root = None

def switch_to_main_button(host_link, port_link):
    global root
    for widget in root.winfo_children():
        widget.destroy()
    
    button = tk.Button(
        root,
        text="Clagnosco запущен. Открыть в браузере",
        bg="#00af09",
        fg="white",
        font=("Segoe UI", 30),
        command=lambda: webbrowser.open(f'http://{host_link}:{port_link}/')
    )
    button.pack()

    webbrowser.open(f'http://{host_link}:{port_link}/')
    
    root.loading_complete = True

def load_and_start_server():
    from webui import run_server
    
    threading.Thread(target=lambda: run_server(HOST_LINK, PORT_LINK,
                                               open_link=False, print_process=False), daemon=True).start()
    print(f"Clagnosco v{PROJECT_VERSION} запущен на {HOST_LINK}:{PORT_LINK}")
    
    time.sleep(1)
    if not IS_COLAB:
        root.after(0, lambda: switch_to_main_button(HOST_LINK, PORT_LINK))

def resource_dir(relative_dir):
    try:
        base_dir = sys._MEIPASS
    except Exception:
        base_dir = os.path.abspath(".")
    return os.path.join(base_dir, relative_dir)

def create_gui_main():
    global root
    root = tk.Tk()
    root.title("Clagnosco")
    root.geometry("750x75")
    root.resizable(False, False)
    icon_dir = resource_dir("webui/static/images/clagnosco.ico")
    root.iconbitmap(icon_dir)
    
    loading_label = tk.Label(
        root,
        text="Загрузка Clagnosco...",
        fg="#000000",
        font=("Segoe UI", 24)
    )
    loading_label.pack(expand=True)
    
    def on_close():
        root.destroy()
        os._exit(0)
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    
    threading.Thread(target=load_and_start_server, daemon=True).start()

    root.mainloop()


if __name__ == '__main__':
    if not IS_COLAB:
        create_gui_main()
    else:
        load_and_start_server()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
