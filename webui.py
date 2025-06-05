from flask import Flask, render_template, send_file, abort, request, jsonify
import os
from PIL import Image
import io
from time import time

from autoencoder import *
from cluster import *


if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

class AppState:
    def __init__(self):
        self.img_dir = None
        self.model_name = None
        self.model = None
        self.cashing = True
        self.img_names = []
        self.img_clusters = []
        self.status = {"status": "idle"}


app = Flask(__name__, static_folder='webui/static', template_folder='webui/templates')
app.state = AppState()

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
    if data['command'] == 'launchProcessing':
        result = launch_processing(data)
    elif data['command'] == 'clusterImages':
        result = cluster_images()
    return jsonify(result)


def launch_processing(data):
    start_time = time()
    state = app.state

    state.img_dir = data['imgDir'].strip('"')
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
            selected_model_dir = os.path.join(SAVE_FOLDER, data['modelName'])
            if os.path.isfile(selected_model_dir):
                try:
                    state.model, _ = model_loader(data['modelName'])
                except:
                    return state_error_model_load(state, data['modelName'], start_time)
            else:
                return state_error_model_load(state, data['modelName'], start_time)

    state.cashing = data['cashing']
    state.status = {"status": "readyToCluster", "time": time() - start_time}
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
                                                cashing=state.cashing,
                                                ignore_errors=True)
    
    state.img_names = [item[0] for item in images_and_latents]
    state.img_clusters = cluster_latent_vectors(images_and_latents)
    cluster_sizes = cluster_measuring(state.img_clusters)
    state.status = {"status": "readyToPopulate",
                    "classSizes": cluster_sizes,
                    "imagesNames": state.img_names,
                    "time": time() - start_time
                    }
    return state.status

def cluster_measuring(clusters):
    # Размеры кластеров
    return [(cluster[0], sum([i[2] for i in cluster[1]])) for cluster in clusters]


if __name__ == '__main__':
    app.run(debug=True)
