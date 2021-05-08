from flask import Flask, render_template, request, jsonify
from PIL import Image
import os, io, sys
import numpy as np
import cv2
import base64

import matplotlib.pyplot as plt

app = Flask(__name__)


@app.route('/editImage', methods=['GET', 'POST'])
def edit_image():
    print(request.files , file=sys.stderr)
    file = request.files['image'].read()  ## byte file
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # color histogram of unedited
    hist_orig = (img.flatten().tolist(), img[0].flatten().tolist(), img[1].flatten().tolist(), img[2].flatten().tolist())

    # image processing
    img += 1

    hist_edit = (img.flatten().tolist(), img[0].flatten().tolist(), img[1].flatten().tolist(), img[2].flatten().tolist())

    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'status': str(img_base64), 'hist_orig': hist_orig, 'hist_edit': hist_edit})


@app.route('/test', methods=['GET', 'POST'])
def test():
    print("log: got at test", file=sys.stderr)
    return jsonify({'status': 'succces'})


@app.route('/')
def home():
    return render_template('image_editor_web.jinja2')


@app.after_request
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
    app.run(debug=True)
