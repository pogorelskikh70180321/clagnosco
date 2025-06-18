# Информация о ВКР "Clagnosco":
#  ФИО автора: Погорельских Константин Владимирович
#  Тема ВКР: «Классификация изображений с помощью искусственного интеллекта (на примере Частного образовательного учреждения высшего образования «Московский университет имени С.Ю. Витте»).»
#  ВУЗ: ЧОУ ВО «Московский университет им. С.Ю. Витте»
#  Специальность: Прикладная информатика [09.03.03] Бакалавр
#  Факультет: Информационных технологий
#  Специализация / Профиль подготовки: Искусственный интеллект и анализ данных
#  Учебная группа: ИД 23.3/Б3-21

from flask import Flask, render_template, send_file, abort, request, jsonify
import logging
import shutil
from datetime import datetime
import time
import gc
import io
import sys
import webbrowser

from autoencoder import *
from cluster import *

import warnings
warnings.filterwarnings('ignore')

PROJECT_VERSION = "1.0.0"

HOST_LINK = '127.0.0.1'
PORT_LINK = 5000


class AppState:
    def __init__(self):
        self.print_process = True
        self.img_dir = None
        self.model_name = None
        self.model = None
        self.caching = True
        self.cluster_number = -1
        self.img_names = []
        self.img_clusters = []
        self.status = {"status": "idle"}
        self.session_id = None

def resource_dir(relative_dir):
    try:
        base_dir = sys._MEIPASS
    except Exception:
        base_dir = os.path.abspath(".")
    return os.path.join(base_dir, relative_dir)

app = Flask(__name__, static_folder=resource_dir('webui/static'), template_folder=resource_dir('webui/templates'))
app.state = AppState()


class ImageRouteFilter(logging.Filter):
    def filter(self, record):
        message_sources = []
        
        if hasattr(record, 'getMessage'):
            try:
                formatted_msg = record.getMessage()
                message_sources.append(formatted_msg)
            except:
                pass
        
        if hasattr(record, 'msg'):
            message_sources.append(str(record.msg))
        
        if hasattr(record, 'msg') and hasattr(record, 'args') and record.args:
            try:
                formatted_with_args = record.msg % record.args
                message_sources.append(str(formatted_with_args))
            except:
                pass
        
        for msg in message_sources:
            if msg and isinstance(msg, str):
                if 'GET /' in msg:
                    try:
                        path_start = msg.find('GET /') + 4
                        path_end = msg.find(' ', path_start)
                        if path_end == -1:
                            path_end = msg.find('"', path_start)
                        if path_end == -1:
                            path_end = len(msg)
                        
                        path = msg[path_start:path_end]
                        
                        if path.startswith('/img/') or path.startswith('/img_small/') or path.startswith('/static/'):
                            return False
                    except:
                        blocked_patterns = ['GET /img/', 'GET /img_small/', 'GET /static/']
                        if any(pattern in msg for pattern in blocked_patterns):
                            return False
        return True


werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.addFilter(ImageRouteFilter())

root_logger = logging.getLogger()
root_logger.addFilter(ImageRouteFilter())


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/img/<path:filename>')
def serve_image(filename):
    full_path = os.path.abspath(filename)
    if not os.path.isfile(full_path):
        return abort(404, description='Изображение не найдено')
    return send_file(full_path)

@app.route('/img_small/<path:filename>')
def serve_image_small(filename, new_size=300, new_quality=85):
    full_path = os.path.abspath(filename)
    if not os.path.isfile(full_path):
        return abort(404, description='Изображение не найдено')
    try:
        with Image.open(full_path) as img:

            img.thumbnail((new_size, new_size))
            img = img.convert('RGB')

            img_io = io.BytesIO()
            img.save(img_io, format='JPEG', quality=new_quality)
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        return abort(500, description=f'Ошибка обработки изображения: {e}')

@app.route('/fetch', methods=['POST'])
def fetch():
    data = request.get_json()
    incoming_session = data["sessionID"]
    state = app.state
    print("Command:", data['command']) if state.print_process else None

    if data['command'] == 'endSession':
        state.session_id = None
    elif data['command'] in ['modelsInFolder', 'clearCache', 'unloadModel',
                             'basicResponse', 'exitClagnosco', 'currentStatus', 'printProcess']:
        pass
    elif data['command'] in ['launchProcessing', 'importData']:
        state.session_id = incoming_session
    elif incoming_session != state.session_id:
        return jsonify({"status": "oldSession"})

    if data['command'] == 'basicResponse':
        start_time = time.time()
        if state.status == "idle":
            return {}
        
        state.status = {
            "status": "basicResponseSuccess",
            "time": time.time() - start_time
            }
        return state.status
    elif data['command'] == 'launchProcessing':
        result = launch_processing(data)
    elif data['command'] == 'clusterImages':
        result = cluster_images()
    elif data['command'] == 'clusterImagesFake':
        result = cluster_images_fake()
    elif data['command'] == 'imageProbsGet':
        result = image_probs_get(data)
    elif data['command'] == 'clearCache':
        result = clear_cache_webui(data)
    elif data['command'] == 'modelsInFolder':
        result = models_in_folder()
    elif data['command'] == 'addEmptyClagnoscoClass':
        result = add_empty_clagnosco_class()
    elif data['command'] == 'createRestClagnoscoClass':
        result = create_rest_clagnosco_class()
    elif data['command'] == 'renameClagnoscoClass':
        result = rename_clagnosco_class(data)
    elif data['command'] == 'copyClagnoscoClass':
        result = copy_clagnosco_class(data)
    elif data['command'] == 'deleteClagnoscoClass':
        result = delete_clagnosco_class(data)
    elif data['command'] == 'clagnoscoClassImagesSelectionUpdate':
        result = clagnosco_class_images_selection_update(data)
    elif data['command'] == 'saveFolder':
        result = save_folder(data)
    elif data['command'] == 'saveTable':
        result = save_table()
    elif data['command'] == 'unloadModel':
        result = unload_model()
    elif data['command'] == 'importData':
        result = import_data(data)
    elif data['command'] == 'currentStatus':
        result = {
            "print_process": state.print_process,
            "img_dir": state.img_dir,
            "model_name": state.model_name,
            "caching": state.caching,
            "cluster_number": state.cluster_number,
            "img_names": state.img_names,
            "img_clusters": state.img_clusters,
            "status": state.status,
            "session_id": state.session_id,
        }
    elif data['command'] == 'endSession':
        result = {"status": "sessionEnded"}
    elif data['command'] == 'printProcess':
        result = print_process_change(data)
    elif data['command'] == 'exitClagnosco':
        os._exit(0)
    else:
        print(f"Неизвестный запрос:\n{data}") if state.print_process else None
    return jsonify(result)

def clear_cache_webui(data):
    start_time = time.time()
    state = app.state
    try:
        state.img_dir = data['imgDir'].strip('"')
        
        if state.img_dir[-1] not in ["\\", "/"]:
            state.img_dir = state.img_dir + "\\"
        clear_cache(state.img_dir)
        state.status = {"status": "cacheCleared", "time": time.time() - start_time}
        return state.status
    except:
        state.status = {
            "status": "error",
            "type": "Cache deletion issue",
            "message": f"Возникла ошибка при удалении кэша",
            "time": time.time() - start_time
            }
        return state.status


def launch_processing(data):
    start_time = time.time()
    state = app.state

    try:
        state.img_dir = data['imgDir'].strip('"')
        # if state.img_dir[-1] not in ["\\", "/"]:
        #     state.img_dir = state.img_dir + "\\"
        
        if not os.path.isdir(state.img_dir):
            state.status = {"status": "error",
                            "type": "Local folder not found",
                            "message": f"Папка \"{state.img_dir}\" не найдена"}
            state.img_dir = None
            return state.status

        if data['modelName'] != state.model_name:
            state.model_name = data['modelName']
            if data['modelName'] == "download":
                state.model, _ = model_loader("download", print_process=state.print_process)
            elif data['modelName'] == "download-save":
                state.model, _ = model_loader("download-save", print_process=state.print_process)
                state.model_name = "model.pt"
            else:
                if not os.path.exists(SAVE_FOLDER):
                    os.makedirs(SAVE_FOLDER)

                selected_model_dir = os.path.join(SAVE_FOLDER, data['modelName'])
                if os.path.isfile(selected_model_dir):
                    try:
                        state.model, _ = model_loader(data['modelName'], print_process=state.print_process)
                    except Exception as e:
                        return state_error_model_load(state, data['modelName'], start_time, error=e)
                else:
                    return state_error_model_load(state, data['modelName'], start_time, error=e)

        state.caching = data['caching']
        state.cluster_number = data['clusterNumber']
        state.status = {"status": "readyToCluster",
                        "time": time.time() - start_time}
        return state.status
    except Exception as e:
        state.status = {
            "status": "error",
            "type": "Launch error",
            "message": f"Ошибка запуска: {e}",
            "time": time.time() - start_time
            }
        return state.status


def state_error_model_load(state, name, start_time, error):
    state.img_dir = None
    unload_model()

    state.status = {
        "status": "error",
        "type": "Local model not found",
        "message": f"Ошибка загрузки модели \"{name}\": {str(error)}",
        "time": time.time() - start_time
        }
    return state.status

def cluster_images():
    start_time = time.time()
    state = app.state
    state.img_clusters = []
    images_and_latents, _, _ = images_to_latent(image_folder=state.img_dir,
                                                model=state.model,
                                                caching=state.caching,
                                                ignore_errors=True,
                                                print_process=state.print_process)
    print("images_to_latent завершено") if state.print_process else None

    if len(images_and_latents) < 2:
        state.status = {
            "status": "error",
            "type": "Too few images",
            "message": f"Было найдено данное количество изображений: {len(images_and_latents)}. Требуется как минимум 2.",
            "time": time.time() - start_time
            }
        return state.status
    
    state.img_names = [item[0] for item in images_and_latents]
    state.img_clusters = cluster_latent_vectors(images_and_latents, cluster_amount=state.cluster_number,
                                                print_process=state.print_process)
    print("cluster_latent_vectors завершено") if state.print_process else None
    cluster_sizes = cluster_measuring(state.img_clusters)
    print("cluster_measuring завершено") if state.print_process else None
    state.status = {"status": "readyToPopulate",
                    "classesSizes": cluster_sizes,
                    "imagesNames": state.img_names,
                    "imagesFolder": state.img_dir,
                    "time": time.time() - start_time
                    }
    return state.status

def cluster_measuring(clusters):
    # Размеры кластеров
    return [(cluster[0], sum([i[2] for i in cluster[1]])) for cluster in clusters]

def image_probs_get(data):
    start_time = time.time()
    state = app.state

    state.status = {"status": "imagesProbs",
                    "probs": state.img_clusters[data['id']][1],
                    "time": time.time() - start_time
                    }
    return state.status

def cluster_images_fake():
    start_time = time.time()
    state = app.state
    
    cluster_sizes = cluster_measuring(state.img_clusters)
    state.status = {"status": "readyToPopulate",
                    "classesSizes": cluster_sizes,
                    "imagesNames": state.img_names,
                    "imagesFolder": state.img_dir,
                    "time": time.time() - start_time
                    }
    return state.status

def models_in_folder():
    start_time = time.time()
    state = app.state

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    state.status = {"status": "readyToInit",
                    "modelNames": [f for f in os.listdir(SAVE_FOLDER) if f.endswith('.pt')],
                    "projectVersion": PROJECT_VERSION,
                    "time": time.time() - start_time
                    }
    return state.status

def add_empty_clagnosco_class():
    start_time = time.time()
    state = app.state
    
    try:
        empty_clagnosco_class = ("Пустой класс", tuple((i, 0.0, False) for i in state.img_names))
        state.img_clusters.append(empty_clagnosco_class)

        state.status = {"status": "emptyClagnoscoClassAdded",
                        "time": time.time() - start_time
                        }
        return state.status
    except:
        state.status = {
            "status": "error",
            "type": "Adding empty class error",
            "message": f"Ошибка добаления пустого класса",
            "time": time.time() - start_time
            }
        return state.status

def create_rest_clagnosco_class():
    start_time = time.time()
    state = app.state
    
    try:
        exclude = set()
        for cluster in state.img_clusters:
            for image_prob in cluster[1]:
                if image_prob[2]:
                    exclude.add(image_prob[0])
        rest_images = [i for i in state.img_names if i not in exclude]

        rest_images_list = [(i, 1.0, True) for i in sorted(rest_images)]
        exclude_list = [(i, 0.0, False) for i in sorted(exclude)]


        rest_clagnosco_class = ("Класс из оставшихся элементов", tuple(rest_images_list + exclude_list))
        state.img_clusters.append(rest_clagnosco_class)

        state.status = {"status": "restClagnoscoClassCreated",
                        "name": rest_clagnosco_class[0],
                        "size": len(rest_images_list),
                        "time": time.time() - start_time
                        }
        return state.status
    except:
        state.status = {
            "status": "error",
            "type": "Adding empty class error",
            "message": f"Ошибка добаления пустого класса",
            "time": time.time() - start_time
            }
        return state.status

def rename_clagnosco_class(data):
    start_time = time.time()
    state = app.state
    
    try:
        clagnosco_class_id = data["id"]
        old_name = state.img_clusters[clagnosco_class_id][0]
        new_name = data["newName"].strip()
        state.img_clusters[clagnosco_class_id] = (new_name, state.img_clusters[clagnosco_class_id][1])

        state.status = {"status": "clagnoscoClassRenamed",
                        "oldName": old_name,
                        "newName": new_name,
                        "time": time.time() - start_time
                        }
        return state.status
    except:
        state.status = {
            "status": "error",
            "type": "Renaming class error",
            "message": f"Ошибка переименования класса",
            "time": time.time() - start_time
            }
        return state.status

def copy_clagnosco_class(data):
    start_time = time.time()
    state = app.state
    
    try:
        clagnosco_class_id = data["id"]
        clagnosco_class_copy = (data["newName"], state.img_clusters[clagnosco_class_id][1])
        state.img_clusters.append(clagnosco_class_copy)

        state.status = {"status": "clagnoscoClassCopied",
                        "time": time.time() - start_time
                        }
        return state.status
    except:
        state.status = {
            "status": "error",
            "type": "Copying class error",
            "message": f"Ошибка копирования класса",
            "time": time.time() - start_time
            }
        return state.status

def delete_clagnosco_class(data):
    start_time = time.time()
    state = app.state
    
    try:
        clagnosco_class_id = data["id"]
        state.img_clusters.pop(clagnosco_class_id)

        state.status = {"status": "clagnoscoClassDeleted",
                        "time": time.time() - start_time
                        }
        return state.status
    except:
        state.status = {
            "status": "error",
            "type": "Deleting class error",
            "message": f"Ошибка удаления класса",
            "time": time.time() - start_time
            }
        return state.status

def clagnosco_class_images_selection_update(data):
    start_time = time.time()
    state = app.state
    
    try:
        clagnosco_class_id = data["id"]

        current_cluster = state.img_clusters[clagnosco_class_id][1]

        updated_cluster = []
        for img_name_webui, selected_webui in data["selection"]:
            for img_name_server, prob_server, selected_server in current_cluster:
                if img_name_webui == img_name_server:
                    updated_cluster.append((img_name_server, prob_server, selected_webui))
                    break
        
        updated_cluster = tuple(updated_cluster)

        state.img_clusters[clagnosco_class_id] = (state.img_clusters[clagnosco_class_id][0], updated_cluster)
        

        state.status = {"status": "clagnoscoClassImagesSelectionUpdated",
                        "time": time.time() - start_time
                        }
        return state.status
    except:
        state.status = {
            "status": "error",
            "type": "Updating selections class error",
            "message": f"Ошибка обновления выбора в классе",
            "time": time.time() - start_time
            }
        return state.status

def save_folder(data):
    start_time = time.time()
    state = app.state

    def name_url(name):
        replacements = {
            '<': r'%3C',
            '>': r'%3E',
            ':': r'%3A',
            '"': r'%22',
            '/': r'%2F',
            '\\': r'%5C',
            '|': r'%7C',
            '?': r'%3F',
            '*': r'%2A',
            '%': r'%25',
        }
        for symbol, url in replacements.items():
            name = name.replace(symbol, url)
        return name
    
    def naming(naming_type, n, clagnosco_class_name, previous_names):

        folder_name = ""
        
        if naming_type == "numberName":
            folder_name = f"{str(n + 1)} - {name_url(clagnosco_class_name.strip())}"
            folder_name = folder_name[:100]
            folder_name = folder_name.strip()
            return folder_name
        elif naming_type == "number":
            folder_name = str(n + 1)
            return folder_name
        elif naming_type == "name":
            folder_name = name_url(clagnosco_class_name.strip())
            if folder_name == "":
                folder_name = "_"
            
            folder_name = folder_name[:100]
            folder_name = folder_name.strip()
            
            original_name = folder_name
            current_n = 2
            if folder_name.lower() in previous_names:
                while True:
                    if folder_name.lower() in previous_names:
                        folder_name = f"{original_name} {current_n}"
                        current_n += 1
                    else:
                        break
            return folder_name
        else:
            return f"error {n + 1}"
        
    try:
        naming_type = data['namingType']
        previous_names = []
        for n, (clagnosco_class_name, clagnosco_class) in enumerate(state.img_clusters):
            subfolder_name = naming(naming_type, n, clagnosco_class_name, previous_names)
            previous_names.append(subfolder_name.lower())

            new_dir = os.path.join(state.img_dir, subfolder_name)
            os.makedirs(new_dir, exist_ok=True)

            for img, _, _ in [i for i in clagnosco_class if i[2]]:
                shutil.copy2(os.path.join(state.img_dir, img),
                             os.path.join(new_dir, img))

        state.status = {"status": "saveFolderSuccess",
                        "folder": state.img_dir,
                        "time": time.time() - start_time}
        return state.status
    except Exception as e:
        state.status = {
            "status": "error",
            "type": "Saving folder error",
            "message": f"Возникла ошибка распределения изображений по субдиректориям в папке «{state.img_dir}»",
            "console": f"Ошибка: {e.__class__.__name__}: {str(e)}",
            "time": time.time() - start_time
        }
        return state.status


def save_table():
    start_time = time.time()
    state = app.state
    try:
        img_dir = state.img_dir
        table_columns = ["ID класса", "Имя класса", "Изображение", "Вероятность", "Входит в класс", f"Директория - {img_dir}"]
        table_list = []

        for clagnosco_class_n, (clagnosco_class_name, clagnosco_class) in enumerate(state.img_clusters):
            clagnosco_class_n = clagnosco_class_n + 1
            clagnosco_class_name = clagnosco_class_name.strip()
            for img, prob, is_member in clagnosco_class:
                table_list.append([clagnosco_class_n,
                                clagnosco_class_name,
                                img,
                                str(prob).replace(".", ","),
                                1 if is_member else 0,
                                ""])

        df = pd.DataFrame(table_list, columns=table_columns)
        # df_csv = df.to_csv(index=False, encoding="utf-8-sig", sep=";")
        df_csv = df.to_csv(index=False, encoding="utf-8", sep=";")  # encoding="cp1251"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_file_name = f"clagnosco классы {timestamp}.csv"

        state.status = {"status": "saveTableSuccess",
                        "table": df_csv,
                        "fileName": csv_file_name, 
                        "time": time.time() - start_time}
        return state.status
    except:
        state.status = {
            "status": "error",
            "type": "Saving table error",
            "message": f"Возникла ошибка создания таблицы",
            "time": time.time() - start_time
        }
        return state.status

def unload_model():
    start_time = time.time()
    state = app.state
    try:
        message = "Нет модели в памяти"
        if state.model is not None:
            try:
                model_device = next(state.model.parameters()).device
            except:
                model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            del state.model
            state.model = None

            gc.collect()

            if model_device.type.lower().startswith("cuda"):
                torch.cuda.set_device(model_device)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            message = "Модель успешно выгружена из памяти"
        state.model_name = None
        state.status = {"status": "modelUnloaded",
                        "message": message,
                        "time": time.time() - start_time}
        return state.status
    except:
        state.status = {
            "status": "error",
            "type": "Model unloading error",
            "message": f"Ошибка выгрузки модели",
            "time": time.time() - start_time
        }
        return state.status

def import_data(data):
    start_time = time.time()
    state = app.state
    try:
        csv_data = data["table"]

        df = pd.read_csv(io.StringIO(csv_data), sep=";", encoding="utf-8")
        df = df.fillna("")

        if not isinstance(csv_data, str):
            raise ValueError(f"Ожидалась строка CSV, но получен тип {type(csv_data).__name__}")
        
        img_dir_column = [col for col in df.columns if col.startswith("Директория - ")]
        if not img_dir_column:
            raise ValueError("Не найдена колонка с директорией в виде: \"Директория - путь\\к\\папке\")")
        
        img_dir = img_dir_column[0].replace("Директория - ", "", 1).strip()
        
        if not os.path.isdir(img_dir):
            raise ValueError(f"Директория не существует: {img_dir}")
        
        state.img_dir = img_dir


        required_columns = {"ID класса", "Имя класса", "Изображение", "Вероятность", "Входит в класс"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Некорректный формат CSV-файла. Нужны все столбцы: {str(required_columns)}")
        df["Вероятность"] = df["Вероятность"].str.replace(",", ".").astype(float)
        
        clusters = []
        for class_id in sorted(df["ID класса"].unique()):
            class_rows = df[df["ID класса"] == class_id]
            class_name = class_rows["Имя класса"].iloc[0]
            cluster = []

            for _, row in class_rows.iterrows():
                img = row["Изображение"]
                prob = row["Вероятность"]
                is_member = bool(int(row["Входит в класс"]))
                cluster.append((img, prob, is_member))

            clusters.append((class_name, cluster))

        state.img_clusters = clusters


        state.status = {
            "status": "dataImported",
            "imgDir": img_dir,
            "time": time.time() - start_time
        }
        return state.status
    
    except Exception as e:
        print(f"Ошибка импорта: {str(e)}") if state.print_process else None
        state.status = {
            "status": "error",
            "type": "Import error",
            "message": f"Ошибка импорта: {str(e)}",
            "time": time.time() - start_time
        }
        return state.status

def print_process_change(data):
    start_time = time.time()
    state = app.state
    try:
        old_print_process = state.print_process
        if "printProcess" in data:
            if not isinstance(data["printProcess"], bool):
                raise ValueError("printProcess должен быть bool")
            state.print_process = data["printProcess"]
            state.status = {"status": "printProcessChanged",
                            "changeType": "manual",
                            "oldState": old_print_process,
                            "newState": state.print_process,
                            "time": time.time() - start_time}
        else:
            state.print_process = not state.print_process
            state.status = {"status": "printProcessChanged",
                            "changeType": "toggle",
                            "oldState": old_print_process,
                            "newState": state.print_process,
                            "time": time.time() - start_time}
        return state.status
    except Exception as e:
        state.status = {
            "status": "error",
            "type": "Print process change error",
            "message": f"Ошибка переключения печати в консоль: {str(e)}",
            "time": time.time() - start_time
        }
        return state.status


def run_server(host_link='127.0.0.1', port_link=5000,
               debug=False, use_reloader=False, open_link=True,
               print_process=True):
    app.state.print_process = print_process
    if open_link:
        webbrowser.open(f'http://{host_link}:{port_link}/')
    app.run(host=host_link, port=port_link, debug=debug, use_reloader=use_reloader)

if __name__ == '__main__':
    run_server(HOST_LINK, PORT_LINK, debug=True, use_reloader=True, open_link=False, print_process=True)
