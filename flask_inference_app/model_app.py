from flask import Blueprint, Flask, request, render_template, flash, redirect
import os
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='flask_inference_app/model/best.pt',force_reload=True) 
bp = Blueprint('routes', __name__, url_prefix='/')

UPLOAD_FOLDER = '/home/lolitav/projects_v3/objects_detection/flask_inference_app/uploads'
RESULTS_FOLDER = '/home/lolitav/projects_v3/objects_detection/flask_inference_app/static'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


@bp.route('/image_process', methods=['POST'])
def image_process():
    print(os.getcwd())
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            file.save(os.path.join(UPLOAD_FOLDER, file.filename))
            result = model(os.path.join(UPLOAD_FOLDER, file.filename))
            result.save(save_dir=RESULTS_FOLDER, exist_ok=True)
            
        return render_template("image_process_output.html", task_name='Prediction Output', user_image=os.path.join('/static', file.filename))

@bp.route('/')
def start_page():
    return render_template('home.html', start_page='True')

@bp.route('/custom_task')
def custom_task():
    return render_template('image_input_form.html', task_name='Uno cards detection - Yolov5')

class Config(object):
    SECRET_KEY = 'YOUR_SECRET_KEY'
    DATABASE = os.path.join('instance', 'project.sqlite')

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config())
    os.makedirs(app.instance_path, exist_ok=True)
    app.register_blueprint(bp)
    print('SERVER READY')
    return app

app = create_app()


if __name__ == '__main__':
    app.run()


