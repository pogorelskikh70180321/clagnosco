from flask import Flask, render_template, send_file, abort, request, jsonify
import os
from PIL import Image
import io

MODELS_FOLDER = 'models'
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)


app = Flask(__name__, static_folder='webui/static', template_folder='webui/templates')

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

@app.route('/api', methods=['POST'])
def api_exchange():
    pass

if __name__ == '__main__':
    app.run(debug=True)
