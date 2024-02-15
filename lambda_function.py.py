import json
import numpy as np
import cv2
import joblib
import base64
from flask import Flask, jsonify, request
import os

def decode(encoded_img):
    aux_path = 'tmp.png'
    with open(aux_path, "wb") as f:
        f.write(encoded_img)
        f.close()

    out = cv2.imread(aux_path)
    if os.path.isfile(aux_path):
        os.remove(aux_path)

    return out
    

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def api():
    #img = decode(request.json)
    if 'image' not in request.files:
        print('first if')
        return "No image"
    file=request.files['image']
    if file.filename == '':
        print("second if")
        return 'No selected file' 
    if file:
        print('inside')
        file2 = request.files.get('image')
        print(file2)
        img_bytes = file2.read()
        img_path = "../test.jpg"
        with open(img_path, "wb") as img:
            img.write(img_bytes)
            img.close()
        img = cv2.imread(img_path)
        #img = decode(file)
        new_array = cv2.resize(img,(100,100))
        new_array = np.array(new_array).reshape(1,-1)
        new_array = new_array/255.0
        model = joblib.load('svcModel.joblib')
        response = model.predict(new_array)
        print(response[0])
        return jsonify(
            body= str(response[0])
        )
    
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')