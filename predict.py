import numpy as np
from keras.models import load_model

from deeplab import run_deeplab

# Load measurement model
measurement_model = load_model('models/measurement_model.h5')
# Load neck prediction model
neck_prediction_model = load_model('models/neck_prediction_model.h5')

def predict_bodyfat(neck, height, waist, gender, hip):
    if gender == 'male':
        bodyfat_percentage = 495 / (1.0324 - 0.19077 * np.log10(waist - neck) + 0.15456 * np.log10(height)) - 450
    else:
        bodyfat_percentage = 495 / (1.29579 - 0.35004 * np.log10(waist + hip - neck) + 0.22100 * np.log10(height)) - 450

    return bodyfat_percentage

def predict(image, gender, height, weight, age): # (imagefile, string, float, float)
    # Prepare ghw data
    if gender == 'male':
        ghw = [0, 1, height, weight]
    else:
        ghw = [1, 0, height, weight]

    # Predict silhouette
    seg_map = run_deeplab(image, 512, 512)
    person_mask = (seg_map == 15).astype(np.uint8)                          # only take the person mask (shape: (256, 256))

    # Prepare data for measurement_model
    mask_pred = np.stack([person_mask, person_mask, person_mask], axis=-1)  # convert to RGB-like (shape: (256, 256, 3))
    mask_pred = np.array(mask_pred)
    mask_pred = np.expand_dims(mask_pred, axis=0)
    ghw = np.array(ghw)
    ghw = np.expand_dims(ghw, axis=0)
    # Predict measuremets and get the first result (index 0)
    measurements = measurement_model.predict([mask_pred, ghw])[0]           # [ankle, arm-length, bicep, calf, chest, forearm, height, hip, leg-length, shoulder-breadth, shoulder-to-crotch, thigh, waist, wrist] 14 total
    
    # Predict neck size
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
    measurements[6] = neck_pred.item()

    # Estimate bodyfat
    bodyfat_pred = predict_bodyfat(neck_pred, height, measurements[12], gender, measurements[7])
    # Add the predicted bodyfat to measurements
    measurements = np.append(measurements, bodyfat_pred)

    return measurements.tolist()