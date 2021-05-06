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
    ######### Do preprocessing here ################
    # color histogram of unedited
    hist_orig = (img[0].flatten().tolist(), img[1].flatten().tolist(), img[2].flatten().tolist())

    img[img > 150] = 0

    # color histogram of edited
    r_edit = cv2.calcHist([img], [0], None, [256], [0,256])
    g_edit = cv2.calcHist([img], [1], None, [256], [0,256])
    b_edit = cv2.calcHist([img], [2], None, [256], [0,256])

    hist_edit = (img[0].flatten().tolist(), img[1].flatten().tolist(), img[2].flatten().tolist())

    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(r_orig)
    # ax[1].plot(r_edit)
    # ax[0].plot(g_orig)
    # ax[1].plot(g_edit)
    # ax[0].plot(b_orig)
    # ax[1].plot(b_edit)
    # plt.show()
    ## any random stuff do here
    ################################################
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
