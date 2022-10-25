FROM python:3.8-slim-buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt

EXPOSE $PORT

#CMD ["python", "yolov5-flask/webapp.py", "--port=5011"]

CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT webapp:app