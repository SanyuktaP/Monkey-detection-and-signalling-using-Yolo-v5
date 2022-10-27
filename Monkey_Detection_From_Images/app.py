import argparse
import io
import os
import pathlib
import json
import numpy as np


from PIL import Image


import torch
from flask import Flask, render_template, request, redirect, send_file

app = Flask(__name__)


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
        model = torch.hub.load('yolov5', 'custom', 'best.pt', source='local')
        results = model([img])
        print('result is',results)
    
        print(results.pandas().xyxy[0])


        results.render()  # updates results.imgs with boxes and labels
        results.save()
        #coords=results.xyxy[0].cpu().numpy()
        #with open("./results.json", "w+") as out_file:
        #    json.dump(coords, out_file, cls=NumpyEncoder)
        
        image_dir = './runs/detect/'
        last_folder = sorted(pathlib.Path(image_dir).glob('*/'), key=os.path.getmtime)[-1]
        print("last_folder: ", last_folder)
        return send_file(f"{last_folder}/image0.jpg")

    return render_template("index.html")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    
    #model.eval()
    app.run(host="0.0.0.0", port=args.port, debug=True)  # debug=True causes Restarting with stat


# from fastapi import FastAPI, File
# from segmentation import get_yolov5, get_image_from_bytes
# from starlette.responses import Response
# import io
# from PIL import Image
# import json
# from fastapi.middleware.cors import CORSMiddleware
# model = get_yolov5()
# app = FastAPI(
#     title="YOLOV5 Monkey Detection API",
#     description="""Obtain object value out of image
#     and return image and json result""",
#     version="0.0.1",
# )
# origins = [
#     "http://localhost",
#     "http://localhost:8000",
#     "*"
# ]
# app.add_middleware(
#      CORSMiddleware,
#      allow_origins=origins,
#      allow_credentials=True,
#      allow_methods=["*"],
#      allow_headers=["*"],
# )

# @app.get('/notify/v1/health')
# def get_health():
#     return dict(msg='OK')

# @app.post("/object-to-json")
# async def detect_food_return_json_result(file: bytes = File(...)):
#     input_image = get_image_from_bytes(file)
#     results = model(input_image)
#     detect_res = results.pandas().xyxy[0].to_json(orient="records")
#     detect_res = json.loads(detect_res)
#     return {"result": detect_res}

# @app.post("/object-to-img")
# async def detect_food_return_base64_img(file: bytes = File(...)):
#     input_image = get_image_from_bytes(file)
#     results = model(input_image)
#     results.render()  # updates results.imgs with boxes and labels
#     for img in results.imgs:
#         bytes_io = io.BytesIO()
#         img_base64 = Image.fromarray(img)
#         img_base64.save(bytes_io, format="jpeg")
#     return Response(content=bytes_io.getvalue(),
# media_type="image/jpeg")