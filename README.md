# Monkey-detection-and-signalling-using-Yolo-v5


### Step 1: Clone my repo
```bash
git clone https://github.com/SanyuktaP/Monkey-detection-and-signalling-using-Yolo-v5.git
```
### Step 2: Create Conda environment

```bash
conda create -p venv python==3.7.6 -y
```
```bash
conda activate venv/
```

### Step 3: Install the requirements
```bash
pip3 install r requirements.txt
```

### Step 4: Already trained from Yolov5 -l model- Can directly use the best.pt for you detection purpose. 
##### Else whole process is coded in Monkey_Detection.ipynb

### Step 5 : Now create a flask app with desired ui
run using:
```bash
python appname.py
```

### Step 6 : Dockerize the files and then create a ci/cd pipeline and deploy it in heroku.

### For Client api:
```bash
python restapi.py --port 5556
```
```bash
curl -X POST -F image=@monkey2.jpg http://127.0.0.1:5556
```

### Heroku app links:
#### For Image detection:  https://monkey-detect-yolov5-image.herokuapp.com/
#### For video Detection:  https://monkey-detect-yolov5-video.herokuapp.com/


