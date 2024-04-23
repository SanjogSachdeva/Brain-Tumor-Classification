from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import cv2
import os

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__, static_url_path='/static')

upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']

            if image.filename == '':
                return 'No selected file'

            if image and allowed_file(image.filename):
                filename = secure_filename(image.filename)
                file_path = os.path.join(app.config['UPLOAD'], filename)
                image.save(file_path)

                dec = {0: 'Glioma Tumor', 1: 'Meningioma Tumor', 2: 'No Tumor', 3: 'Pituitary Tumor'}

                selected_image = cv2.imread(file_path)
                selected_image = cv2.resize(selected_image, (128, 128))
                selected_image = selected_image / 255.0

                selected_image = selected_image.reshape(1, 128, 128, 3)
                predictions = model.predict(selected_image)
                predicted_class_index = np.argmax(predictions)

                return render_template('index.html', file_path=file_path, prediction=dec[predicted_class_index], section_to_scroll='scrollToSection')
            else:
                return 'Invalid file format. Allowed formats: png, jpg, jpeg'
        else:
            return 'No file uploaded'
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
