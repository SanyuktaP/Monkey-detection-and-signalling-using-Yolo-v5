FROM python:3.8
ADD . /app
WORKDIR /app

RUN pip install -r requirements.txt
EXPOSE $PORT

CMD ["python", "yolov5-flask/webapp.py", "--port=5000"]
