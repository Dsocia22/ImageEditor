from flask import Flask, render_template, request, jsonify
from PIL import Image
import os, io, sys
import numpy as np
import torch
import base64

import matplotlib.pyplot as plt

import GAN
# load model
path = r'./gan_model_trained.pth15'

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

gen = GAN.Generator().to(device)
state = torch.load(path, map_location=lambda storage, loc: storage)
state_dict = {k[7:]: v for k, v in state["generator_state_dict"].items()}
gen.load_state_dict(state_dict)


app = Flask(__name__)


@app.route('/editImage', methods=['GET', 'POST'])
def edit_image():
    print(request.files , file=sys.stderr)
    file = request.files['image'].read()  ## byte file
    image = np.array(Image.open(io.BytesIO(file)))[:, :, :3]
    # image = image[:, :, ::-1].copy()
    #plt.imshow(image)
    #plt.show()
    # convert from numpy to torch
    image = torch.from_numpy(image).permute(2, 0, 1)[None, :, :, :].to(device).float() / 255
    
        # downsample image
    (h, w) = image.shape[2:]
    size = 756
    if h >= w:
        r = size / float(h)
        dim = (size, int(w * r))
    else:
        r = size / float(w)
        dim = (int(h * r), size)

    image = torch.nn.functional.interpolate(image, dim)

    img = image.cpu().detach().numpy()[0, :, :, :] * 255

    # color histogram of unedited
    hist_orig = (img.flatten().tolist(), img[0].flatten().tolist(), img[1].flatten().tolist(), img[2].flatten().tolist())

    # image processing
    
    image = gen.forward(image)
    image = image.permute(0, 2, 3, 1)
    image = image.cpu().detach().numpy()[0, :, :, :] * 255
    mask = image < 0
    image[mask] = 0
    mask = image > 255
    image[mask] = 255

    hist_edit = (image.flatten().tolist(), image[:, :, 0].flatten().tolist(), image[:, :, 1].flatten().tolist(), image[:, :, 2].flatten().tolist())

    img = Image.fromarray(image.astype("uint8"))
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
    app.run(debug=True, port='5000', host='0.0.0.0')
