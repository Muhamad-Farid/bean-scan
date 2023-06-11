from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
from google.cloud import storage

app = Flask(__name__)
model = load_model('TDCNN.h5')

# Menginisialisasi client storage Google Cloud
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credentials.json'
storage_client = storage.Client()
bucket_name = 'scanmachinelearning'

# Function to load and prepare the image in the right shape
def read_image(filename):
    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Mengecek apakah ekstensi file diizinkan
def allowed_file(filename):
    allowed_ext = ['jpg', 'jpeg', 'png']
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext

# Menyimpan file ke bucket di Google Cloud Storage
def save_file_to_bucket(file, filename, user_id):
    bucket = storage_client.bucket(bucket_name)
    folder_name = f'user_{user_id}'
    blob_name = os.path.join(folder_name, filename)
    blob = bucket.blob(blob_name)
    with open(file, "rb") as f:
        blob.upload_from_file(f)
    return blob.public_url

@app.route('/')
def index_view():
    return 'Server is running'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)
            class_prediction = model.predict(img)
            classes_x = np.argmax(class_prediction, axis=1)
            if classes_x == 0:
                labels = "13%"
                pesan = "pesan kualitas 1"
            elif classes_x == 1:
                labels = "13.2%"
                pesan = "pesan kualitas 2"
            elif classes_x == 2:
                labels = "14%"
                pesan = "pesan kualitas 3"
            elif classes_x == 3:
                labels = "15%"
                pesan = "pesan kualitas 4"
            else:
                labels = "> 15%"
                pesan = "pesan kualitas 5"
            
            # Simpan file ke bucket di Google Cloud Storage dengan folder berdasarkan user_id
            blob_url = save_file_to_bucket(file_path, filename, user_id)

            result = {
                'labels': labels,
                'pesan_kualitas': pesan,
                'user_image': blob_url
            }

            return jsonify(result)
        else:
            return jsonify({'error': 'Unable to read the file. Please check the file extension'})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8080)

