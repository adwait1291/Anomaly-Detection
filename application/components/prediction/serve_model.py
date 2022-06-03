from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model
MODEL_PATH = 'application/Assets/anomaly.h5'
model = None


def loadModel():
    model = load_model(MODEL_PATH)
    print("Model loaded")
    return model

def check_anomaly(img):
    reconstruction_error_threshold = 0.0035
    img = np.array(img.resize((224,224), Image.ANTIALIAS))
    img = img / 255.
    img = img[np.newaxis, :,:,:]
    reconstruction = model.predict([[img]])
    reconstruction_error = model.evaluate([reconstruction],[[img]], batch_size = 1)[0]

    if reconstruction_error > reconstruction_error_threshold:
        return "Anomaly" 
    else:
        return "NotAnAnomaly"


def predict(image: Image.Image):
    global model
    if model is None:
        model = loadModel()

    #image = np.asarray(image.resize((224, 224)))[..., :3]
    #image = np.expand_dims(image, 0)
    #image = image / 255.0

    result = check_anomaly(image)
    return result


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image
