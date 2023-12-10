from flask import Flask, request, jsonify
from google.cloud import storage
from google.cloud.storage import Blob

import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import img_to_array

app = Flask(__name__)

# PROJECT_ID = 'equifit-testing'
# BUCKET_NAME = 'equifit-model-bucket'

# client = storage.Client.from_service_account_json('equifit-storage-keyfile.json')

# Load models
silhouette_model = load_model('models/silhouette_model.h5')
measurement_model = load_model('models/measurement_model.h5')

def load_image(image, h=512, w=512):
    image = tf.image.resize_with_pad(image, target_height=h, target_width=w)    # resize with padding so it's not deform the image

    return img_to_array(image)/255.       # return the normalized image array

def predict(image, gender, height, weight): # (imagefile, string, float, float)
    # Prepare data
    image = load_image(image, 256, 256)
    if gender == 'male':
        ghw = [0, 1, height, weight]
    elif gender == 'female':
        ghw = [1, 0, height, weight]
        
    image = np.array(image)
    ghw = np.array(ghw)
    image = np.expand_dims(image, axis=0)
    ghw = np.expand_dims(ghw, axis=0)

    # Predict silhouette
    mask_pred = silhouette_model.predict(image)
    # Predict measuremets
    measurements = measurement_model.predict([mask_pred, ghw])

    return measurements[0].tolist() # [ankle, arm-length, bicep, calf, chest, forearm, height, hip, leg-length, shoulder-breadth, shoulder-to-crotch, thigh, waist, wrist] 14 total

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # if file:
    #     #Nama file di GCS
    #     gcs_file_name = file.filename
    #     #Bucket GCS
    #     bucket = client.bucket(BUCKET_NAME)
    #     # Upload file ke GCS
    #     blob = Blob(gcs_file_name, bucket)
    #     blob.upload_from_file(file)
    #     #URL publik untuk akses file
    #     file_url = f'https://storage.googleapis.com/{BUCKET_NAME}/{gcs_file_name}'

    #     return jsonify({'message': 'File successfully uploaded', 'file_url': file_url})

    if file:
        # Read file content
        image_content = file.read()
        image = tf.image.decode_image(image_content)
        # Get the ghw
        gender = request.form.get('gender')
        height_str = request.form.get('height')
        weight_str = request.form.get('weight')

        if height_str is None or weight_str is None:
            return jsonify({'error': 'Height and weight must be provided as valid numbers'})

        try:
            height = float(height_str)
            weight = float(weight_str)
        except ValueError:
            return jsonify({'error': 'Invalid height or weight format'})
        
        # Predict
        predictions = predict(image, gender, height, weight)
        predictions_dict = {
            'ankle': predictions[0],
            'arm-length': predictions[1],
            'bicep': predictions[2],
            'calf': predictions[3],
            'chest': predictions[4],
            'forearm': predictions[5],
            'height': predictions[6],
            'hip': predictions[7],
            'leg-length': predictions[8],
            'shoulder-breadth': predictions[9],
            'shoulder-to-crotch': predictions[10],
            'thigh': predictions[11],
            'waist': predictions[12],
            'wrist': predictions[13]
        }
        return jsonify({'predictions': predictions_dict})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
