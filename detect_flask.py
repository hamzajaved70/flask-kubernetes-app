# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
from flask import Flask, request, Response, jsonify
import io as StringIO
import base64
from io import BytesIO
import io
import json
from PIL import Image

confthres = 0.3
nmsthres = 0.1
yolo_path = './'

def get_labels(labels_path):
    lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net

def get_prediction(image,net,LABELS,COLORS):
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    print(layerOutputs)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confthres:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres, nmsthres)
    labels = []
    accuracy = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            labels.append(LABELS[classIDs[i]])
            accuracy.append(confidences[i])
    results = [{'label': label, 'accuracy': acc} for label, acc in zip(labels, accuracy)]
    return results

labelsPath="yolo_v3/coco.names"
cfgpath="yolo_v3/yolov3-tiny.cfg"
wpath="yolo_v3/yolov3-tiny.weights"
Lables=get_labels(labelsPath)
CFG=get_config(cfgpath)
Weights=get_weights(wpath)
#nets=load_model(CFG,Weights)
Colors=get_colors(Lables)
app = Flask(__name__)

@app.route('/api/object_detection', methods=['POST'])
def main():
    nets=load_model(CFG,Weights)
    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))
    npimg=np.array(img)
    image=npimg.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    res=get_prediction(image,nets,Lables,Colors)
    jsondata = {}
    jsondata["objects"] = res
    json_doc = json.dumps(jsondata)
    return Response(response=json_doc, status=200, mimetype="application/json")

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
