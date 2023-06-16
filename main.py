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
                labels = "<= 12%"
                pesan = "Kadar air biji kopi sudah sangat bagus dan tidak disarankan untuk melakukan pengeringan lagi      karena hal tersebut dapat menyebabkan bobot biji menjadi sangat ringan "
            elif classes_x == 1:
                labels = "13%-16%"
                pesan = "Kadar air 13%-16% tergolong cukup bagus dan anda bisa saja melakukan pengeringan dengan suhu yang tidak lebih dari 50 derajat celcius dengan waktu dua sampai tiga jam. Hal ini untuk menghindari pemanasan berlebih terhadap biji kopi, karena pemanasan berlebih dapat menyebabkan kualitas biji kopi dan bobot biji kopi menjadi menurun."
            elif classes_x == 2:
                labels = "17%-20%"
                pesan = "Secara umum kadar air dengan persentase 17% sampai 20% tergolong masih banyak mengandung air yang artinya perlu dikeringkan lagi. Pengeringan sebiaknya dilakukan menggunakan oven karena hal ini lebih dapat dikontrol dan menghemat biaya pengeringan. Pengeringan sebaiknya dilakukan dengan suhu 50 derajat celcius sampai 60 derajat celcius dengan durasi selama kurang lebih 1 sampai 2 jam."
            elif classes_x == 3:
                labels = "21%-40%"
                pesan = "Kadar air sebesar ini dapat dilakukan pengeringan secara konvensional dan modern hal itu tergantung dari jumlah biji yang akan dikeringkan. Dengan durasi sebesar 20 jam pada suhu 50 derajat celcius sampai 60 derajat celcius maka kadar air sebesar 12% dapat dicapai."
            elif classes_x == 4:
                labels = ">50% -55%"
                pesan = "Kekeringan ini biasanya didapatkan karena biji kopi yang baru dikupas dikeringkan pada suhu kamar dengan udara bebas. Pengeringan dengan kadar air sebesar ini disarankan menggunakan pengeringan konvensional atau tidak memakai oven karena hal ini dapat menyebabkan pemborosan energi dan terbilang tidak efisien jika  biji kopi yang dikeringkan dalam jumlah besar. Namun jika dalam jumlah yang tidak terlalu besar pengeringan dapat dilakukan dengan oven menggunakan dengan suhu di atas 60 derajat celcius dengan durasi sekitar 25 sampai 30 jam."
            else:
                labels = "bukan kopi"
                pesan = "Gambar yang dimasukkan bukan gambar biji kopi."
            
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

