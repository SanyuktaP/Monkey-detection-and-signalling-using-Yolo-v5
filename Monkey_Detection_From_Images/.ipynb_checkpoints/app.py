"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
import pathlib

from PIL import Image

import torch
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename


app = Flask(__name__, template_folder="./templates")
UPLOAD_FOLDER = './templates/static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024



@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])

        results.render()  # updates results.imgs with boxes and labels
        results.save()
        image_dir = './runs/detect/'
        last_folder = sorted(pathlib.Path(image_dir).glob('*/'), key=os.path.getmtime)[-1]
        print("last_folder: ", last_folder)
        return send_file(f"../{last_folder}/image0.jpg")

    return render_template("index.html")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('yolov5', 'custom', 'best.pt', source='local', force_reload=True)
    #model.eval()
    app.run(host="0.0.0.0", debug=False, port=args.port)  # debug=True causes Restarting with stat

