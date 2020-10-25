import cv2
import numpy as np
from flask import Flask, jsonify, request
#from flask_restful import Resource, Api
from PIL import Image
import io #, sys , os

app = Flask(__name__)

@app.route('/api/object_detection', methods=['POST'])
def objectDetection():
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")  #dnn = deep neural network
    classes = []  # for contain all the name from coco file
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]   # classes contain all the name in coco.names file one line is one element in classes list
    layer_names = net.getLayerNames()  #net object
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]   # get the output layer  #until this line is load algorithm
    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))
    npimg=np.array(img)  # source https://medium.com/csmadeeasy/send-and-receive-images-in-flask-in-memory-solution-21e0319dcc1
    img=npimg.copy()  # source https://medium.com/csmadeeasy/send-and-receive-images-in-flask-in-memory-solution-21e0319dcc1
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # source  https://medium.com/csmadeeasy/send-and-receive-images-in-flask-in-memory-solution-21e0319dcc1
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)  #
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    result =[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)  # np.argmax() Returns the indices of the maximum values along an axis.
            confidence = scores[class_id]
            if confidence > 0.5:   # the confidence is run from 0 to 1
                # Object detected
                boxes.append(detection[0])    #how many object detect
                confidences.append(round(float(confidence),2))   # percent of confidence
                class_ids.append(class_id)  # type of objects

    for i in range(len(boxes)):  # box is how many object detected on image
        label = str('object: ' + classes[class_ids[i]] + ' ')  # all name of detected index
        percent = str('percent confidence: ' + str(confidences[i]) + ' ')
        combine = label + percent
        result.append(combine)
    return jsonify(result)


if __name__ == '__main__':
    app.run()


    # https://www.youtube.com/watch?v=s_ht4AKnWZg&t=12s&fbclid=IwAR37yWYM_hs2kLkwaORQP5Lrg1FcJOLUBjv0Zjjrn0w8JHeh2MeGg_JBEcc
