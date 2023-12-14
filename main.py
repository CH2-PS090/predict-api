from flask import Flask, request, jsonify
from google.cloud import storage
from google.cloud.storage import Blob

import tarfile
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import img_to_array

app = Flask(__name__)

# PROJECT_ID = 'equifit-testing'
# BUCKET_NAME = 'equifit-model-bucket'

# client = storage.Client.from_service_account_json('equifit-storage-keyfile.json')

# Load measurement model
measurement_model = load_model('models/measurement_model.h5')
neck_prediction_model = load_model('models/neck_prediction_model.h5')

# Creates and loads pretrained deeplab model
tarball_path = "models/deeplab_model.tar.gz"
graph_def = None
graph = tf.Graph()
# Extract frozen graph from tar archive
tar_file = tarfile.open(tarball_path)
for tar_info in tar_file.getmembers():
    if 'frozen_inference_graph' in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
        break
tar_file.close()
# Check if the inference graph was found
if graph_def is None:
    raise RuntimeError('Cannot find inference graph in tar archive.')
# Create tensorflow session from the imported graph
with graph.as_default():
    tf.import_graph_def(graph_def, name='')
DEEPLAB_SESSION = tf.compat.v1.Session(graph=graph)

def process_image(image, h=512, w=512):
    image = tf.image.resize_with_pad(image, target_height=h, target_width=w)    # resize with padding so it's not deform the image

    return img_to_array(image)

def run_deeplab(image, h=512, w=512):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

    # Preprocess the image
    image_array = process_image(image, h, w)
    # Run inference
    batch_seg_map = DEEPLAB_SESSION.run(OUTPUT_TENSOR_NAME, feed_dict={INPUT_TENSOR_NAME: [image_array]})
    seg_map = batch_seg_map[0]
    return seg_map

def predict_bodyfat(neck, height, waist, is_male=True, hip=None):
    if is_male:
        bodyfat_percentage = 495 / (1.0324 - 0.19077 * np.log10(waist - neck) + 0.15456 * np.log10(height)) - 450
    else:
        if hip is None:
            raise ValueError("Hip measurement is required for females.")
        bodyfat_percentage = 495 / (1.29579 - 0.35004 * np.log10(waist + hip - neck) + 0.22100 * np.log10(height)) - 450

    return bodyfat_percentage

def predict(image, gender, height, weight, age): # (imagefile, string, float, float)
    # Prepare ghw data
    if gender == 'male':
        ghw = [0, 1, height, weight]
    elif gender == 'female':
        ghw = [1, 0, height, weight]
    ghw = np.array(ghw)
    ghw = np.expand_dims(ghw, axis=0)

    # Predict silhouette
    seg_map = run_deeplab(image, 256, 256)
    person_mask = (seg_map == 15).astype(np.uint8)                          # only take the person mask (shape: (256, 256))
    mask_pred = np.stack([person_mask, person_mask, person_mask], axis=-1)  # convert to RGB-like (shape: (256, 256, 3))
    # Predict measuremets
    measurements = measurement_model.predict([mask_pred, ghw])              # [ankle, arm-length, bicep, calf, chest, forearm, height, hip, leg-length, shoulder-breadth, shoulder-to-crotch, thigh, waist, wrist] 14 total
    # Predict neck
    neck_pred = neck_prediction_model.predict([
        age,
        weight,
        height,
        measurements[4],    # chest
        measurements[12],   # waist
        measurements[7],    # hip
        measurements[11],   # thigh
        measurements[0],    # ankle
        measurements[2],    # bicep
        measurements[5],    # forearm
        measurements[13]    # wrist
    ])
    # Replace height prediction with neck prediction
    measurements[6] = neck_pred
    # Estimate bodyfat
    bodyfat_pred = None
    if gender == 'male':
        bodyfat_pred = predict_bodyfat(neck_pred, height, measurements[12])
    else:
        bodyfat_pred = predict_bodyfat(neck_pred, height, measurements[12], False, measurements[7])
    measurements.append(bodyfat_pred)

    return measurements[0].tolist()

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
        age_str = request.form.get('age')

        if height_str is None or weight_str is None or age_str is None:
            return jsonify({'error': 'Height, weight, and age must be provided as valid numbers'})

        try:
            height = float(height_str)
            weight = float(weight_str)
            age    = int(age_str)
        except ValueError:
            return jsonify({'error': 'Invalid height, weight, or age format'})
        
        # Predict
        predictions = predict(image, gender, height, weight, age)
        predictions_dict = {
            'ankle': predictions[0],
            'arm-length': predictions[1],
            'bicep': predictions[2],
            'calf': predictions[3],
            'chest': predictions[4],
            'forearm': predictions[5],
            'neck': predictions[6],
            'hip': predictions[7],
            'leg-length': predictions[8],
            'shoulder-breadth': predictions[9],
            'shoulder-to-crotch': predictions[10],
            'thigh': predictions[11],
            'waist': predictions[12],
            'wrist': predictions[13],
            'bodyfat': predictions[14]
        }
        return jsonify({'predictions': predictions_dict})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
