"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
import io
from PIL import Image

import torch
from flask import Flask, request
import json

app = Flask(__name__)

DETECTION_URL = "/"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

coords = []

@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        model = torch.hub.load('yolov5', 'custom', 'best.pt', source='local')
        results = model([img])
        #results = model(img, size=640)
        results.render()  # updates results.imgs with boxes and labels
        results.save()
        data = results.pandas().xyxy[0].to_json(orient="records")
        with open("results.json", "w") as out_file:
            json.dump(data, out_file, cls = NumpyEncoder)
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()


    #model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat