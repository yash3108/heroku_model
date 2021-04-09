import pandas as pd
import numpy as np
from tensorflow import keras
from flask import Flask, jsonify, request
import cv2

def process(str):
    str = str.replace('[', '')
    str = str.replace(']', '')
    str = str.split(', ')
    lst = list(map(int, str))
    #print(len(lst))
    arr = np.array(lst)
    arr = arr.reshape(200, 200, 3)
    #print(type(arr[0,0][0]), type(img1[0,0][0]))
    #print((img1 == arr).all())
    arr = arr.astype('uint8')
    image = cv2.resize(arr, (64, 64))
    image = np.array(image)
    image = image.astype('float32')/255.0
    image = image.reshape(-1, 64, 64, 3)

# load model
model = keras.models.load_model(ASL1.h5)

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # get processed data
    image = process(data)

    # predictions
    result = model.predict(image)

    # send back to browser
    output = np.argmax(result, axis=1)

    # return data
    return jsonify(output[0])

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
