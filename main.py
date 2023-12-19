from flask import Flask, request, jsonify
from google.cloud import storage
from google.cloud.storage import Blob
import tensorflow as tf
from flask_mysqldb import MySQL
from datetime import datetime
import jwt
from predict import predict

app = Flask(__name__)

app.config['MYSQL_HOST'] = '34.101.224.237'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '5;KQRYYy3iV8.Jz}'
app.config['MYSQL_DB'] = 'equifit'
app.config['SECRET_KEY'] = 'equifit'

mysql = MySQL(app)
today = datetime.today().date()
# PROJECT_ID = 'equifit-testing'
# BUCKET_NAME = 'equifit-model-bucket'

# client = storage.Client.from_service_account_json('equifit-storage-keyfile.json')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Ambil token dari header Authorization
        authorization_header = request.headers.get('Authorization')
        if authorization_header and authorization_header.startswith('Bearer '):
            token = authorization_header.split(' ')[1]
            decoded_token = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            userId = decoded_token.get('userId')
            email = decoded_token.get('email')
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401
            
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
        image = tf.image.decode_image(image_content, channels=3)
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
        cursor = mysql.connection.cursor()
        cursor.execute(''' INSERT INTO history VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',(userId,predictions[0],predictions[1],predictions[2],predictions[3],predictions[4],predictions[5],predictions[6],predictions[7],predictions[8],predictions[9],predictions[10],predictions[11],predictions[12],predictions[13],predictions[14],today))
        mysql.connection.commit()
        cursor.close()
        return jsonify({'predictions': predictions_dict})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
