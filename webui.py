from flask import Flask, render_template, send_file, abort, request, jsonify
import logging
import os
from PIL import Image
import io
from time import time

from autoencoder import *
from cluster import *


class AppState:
    def __init__(self):
        self.img_dir = None
        self.model_name = None
        self.model = None
        self.caching = True
        self.img_names = []
        self.img_clusters = []
        self.status = {"status": "idle"}


app = Flask(__name__, static_folder='webui/static', template_folder='webui/templates')
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
def serve_image_small(filename):
    full_path = os.path.abspath(filename)
    if not os.path.isfile(full_path):
        return abort(404, description='Изображение не найдено')
    try:
        with Image.open(full_path) as img:
            img.thumbnail((250, 250))
            img_io = io.BytesIO()
            img_format = img.format if img.format else 'PNG'
            img.save(img_io, format=img_format)
            img_io.seek(0)
            return send_file(img_io, mimetype=f'image/{img_format.lower()}')
    except Exception as e:
        return abort(500, description=f'Ошибка обработки изображения: {e}')

@app.route('/fetch', methods=['POST'])
def fetch():
    data = request.get_json()
    if data['command'] == 'basicResponse':
        start_time = time()
        state = app.state
        
        state.status = {
            "status": "basicResponseSuccess",
            "time": time() - start_time
            }
        return state.status
    elif data['command'] == 'launchProcessing':
        result = launch_processing(data)
    elif data['command'] == 'clusterImages':
        result = cluster_images()
    elif data['command'] == 'imageProbsGet':
        result = image_probs_get(data)
    elif data['command'] == 'clearCache':
        result = clear_cache_webui(data)
    elif data['command'] == 'modelsInFolder':
        result = models_in_folder()
    elif data['command'] == 'addEmptyClagnoscoClass':
        result = add_empty_clagnosco_class()
    elif data['command'] == 'copyClagnoscoClass':
        result = copy_clagnosco_class(data)
    elif data['command'] == 'deleteClagnoscoClass':
        result = delete_clagnosco_class(data)
    return jsonify(result)

def clear_cache_webui(data):
    start_time = time()
    state = app.state
    try:
        state.img_dir = data['imgDir'].strip('"')
        
        if state.img_dir[-1] not in ["\\", "/"]:
            state.img_dir = state.img_dir + "\\"
        clear_cache(state.img_dir)
        state.status = {"status": "cacheCleared", "time": time() - start_time}
        return state.status
    except:
        state.status = {
            "status": "error",
            "type": "Cache deletion issue",
            "message": f"Возникла ошибка при удалении кэша",
            "time": time() - start_time
            }
        return state.status


def launch_processing(data):
    start_time = time()
    state = app.state

    try:
        state.img_dir = data['imgDir'].strip('"')
        if state.img_dir[-1] not in ["\\", "/"]:
            state.img_dir = state.img_dir + "\\"
        
        if not os.path.isdir(state.img_dir):
            state.status = {"status": "error",
                            "type": "Local folder not found",
                            "message": f"Папка \"{state.img_dir}\" не найдена"}
            state.img_dir = None
            return state.status

        if data['modelName'] != state.model_name:
            state.model_name = data['modelName']
            if data['modelName'] == "download":
                state.model, _ = model_loader("download")
            else:
                if not os.path.exists(SAVE_FOLDER):
                    os.makedirs(SAVE_FOLDER)

                selected_model_dir = os.path.join(SAVE_FOLDER, data['modelName'])
                if os.path.isfile(selected_model_dir):
                    try:
                        state.model, _ = model_loader(data['modelName'])
                    except:
                        return state_error_model_load(state, data['modelName'], start_time)
                else:
                    return state_error_model_load(state, data['modelName'], start_time)

        state.caching = data['caching']
        state.status = {"status": "readyToCluster",
                        "time": time() - start_time}
        return state.status
    except:
        state.status = {
            "status": "error",
            "type": "Launch error",
            "message": f"Ошибка запуска",
            "time": time() - start_time
            }
        return state.status


def state_error_model_load(state, name, start_time):
    state.img_dir = None
    state.model_name = None
    state.model = None
    state.status = {
        "status": "error",
        "type": "Local model not found",
        "message": f"Модель \"{name}\" не найдена",
        "time": time() - start_time
        }
    return state.status

def cluster_images():
    start_time = time()
    state = app.state
    state.img_clusters = []
    images_and_latents, _, _ = images_to_latent(image_folder=state.img_dir,
                                                model=state.model,
                                                caching=state.caching,
                                                ignore_errors=True,
                                                print_process=True)
    print("images_to_latent завершено")

    if len(images_and_latents) < 2:
        state.status = {
            "status": "error",
            "type": "Too few images",
            "message": f"Было найдено данное количество изображений: {len(images_and_latents)}. Требуется как минимум 2.",
            "time": time() - start_time
            }
        return state.status
    
    state.img_names = [item[0] for item in images_and_latents]
    state.img_clusters = cluster_latent_vectors(images_and_latents, print_process=True)
    print("cluster_latent_vectors завершено")
    cluster_sizes = cluster_measuring(state.img_clusters)
    print("cluster_measuring завершено")
    state.status = {"status": "readyToPopulate",
                    "classesSizes": cluster_sizes,
                    "imagesNames": state.img_names,
                    "imagesFolder": state.img_dir,
                    "time": time() - start_time
                    }
    return state.status

def cluster_measuring(clusters):
    # Размеры кластеров
    return [(cluster[0], sum([i[2] for i in cluster[1]])) for cluster in clusters]

def image_probs_get(data):
    start_time = time()
    state = app.state

    state.status = {"status": "imagesProbs",
                    "probs": state.img_clusters[data['id']][1],
                    "time": time() - start_time
                    }
    return state.status

def models_in_folder():
    start_time = time()
    state = app.state

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    state.status = {"status": "readyToInit",
                    "modelNames": [f for f in os.listdir(SAVE_FOLDER) if f.endswith('.pt')],
                    "time": time() - start_time
                    }
    return state.status

def add_empty_clagnosco_class():
    start_time = time()
    state = app.state
    
    try:
        empty_clagnosco_class = ("Пустой класс", tuple((i, 0.0, False) for i in state.img_names))
        state.img_clusters.append(empty_clagnosco_class)

        state.status = {"status": "emptyClagnoscoClassAdded",
                        "time": time() - start_time
                        }
        return state.status
    except:
        state.status = {
            "status": "error",
            "type": "Adding empty class error",
            "message": f"Ошибка добаления пустого класса",
            "time": time() - start_time
            }
        return state.status

def copy_clagnosco_class(data):
    start_time = time()
    state = app.state
    
    try:
        clagnosco_class_id = data["id"]
        clagnosco_class_copy = (data["newName"], state.img_clusters[clagnosco_class_id][1])
        state.img_clusters.append(clagnosco_class_copy)

        state.status = {"status": "clagnoscoClassCopied",
                        "time": time() - start_time
                        }
        return state.status
    except:
        state.status = {
            "status": "error",
            "type": "Copying class error",
            "message": f"Ошибка копирования класса",
            "time": time() - start_time
            }
        return state.status

def delete_clagnosco_class(data):
    start_time = time()
    state = app.state
    
    try:
        clagnosco_class_id = data["id"]
        state.img_clusters.pop(clagnosco_class_id)

        state.status = {"status": "clagnoscoClassDeleted",
                        "time": time() - start_time
                        }
        return state.status
    except:
        state.status = {
            "status": "error",
            "type": "Deleting class error",
            "message": f"Ошибка удаления класса",
            "time": time() - start_time
            }
        return state.status

if __name__ == '__main__':
    app.run(debug=True)
