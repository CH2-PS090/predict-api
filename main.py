from flask import Flask, request, jsonify
from google.cloud import storage
from google.cloud.storage import Blob

app = Flask(__name__)

PROJECT_ID = 'equifit-testing'
BUCKET_NAME = 'equifit-model-bucket'

client = storage.Client.from_service_account_json('equifit-storage-keyfile.json')


@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        #Nama file di GCS
        gcs_file_name = file.filename
        #Bucket GCS
        bucket = client.bucket(BUCKET_NAME)
        # Upload file ke GCS
        blob = Blob(gcs_file_name, bucket)
        blob.upload_from_file(file)
        #URL publik untuk akses file
        file_url = f'https://storage.googleapis.com/{BUCKET_NAME}/{gcs_file_name}'

        return jsonify({'message': 'File successfully uploaded', 'file_url': file_url})

if __name__ == '__main__':
    app.run(debug=True)
