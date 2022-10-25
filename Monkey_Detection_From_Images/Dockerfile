FROM python:3.7-slim-buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app
ADD . /app
#ENV GUNICORN_CMD_ARGS="--chdir=./yolov5-flask/"
RUN pip install -r requirements.txt

#CMD ["python", "yolov5-flask/app.py", "--port=5011"]

EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app --timeout 8000