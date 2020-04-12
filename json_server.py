from flask import Flask, request
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from flask import Flask,request,redirect,jsonify
import os
import urllib.request
from werkzeug.utils import secure_filename
from base64 import b64encode
import json
import pickle
import csv
import numpy as np
import pandas as pd
UPLOAD_FOLDER = '/upload'
# ALLOWED_EXTENSIONS = {'json'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key="secretkey"

@app.route("/")
def hello():
    return "ASL detection application is running smoothly"

@app.route("/predict", methods=['POST','PUT'])
def print_filename():
    request_data = request.get_json()
    filename='test.json'
    with open(filename, 'w') as f:
        json.dump(request_data, f)
    info = json.loads(open(filename).read())
    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
       'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
       'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
       'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
       'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
       'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
       'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
       'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
       'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
    # info = json.loads(open('key_points1.json', 'r').read())  #Put the name of the .json here
    csv = np.zeros((len(info), len(columns)))
    for i in range(csv.shape[0]):
        one = []
        one.append(info[i]['score'])
        for object in info[i]['keypoints']:
            one.append(object['score'])
            one.append(object['position']['x'])
            one.append(object['position']['y'])
        csv[i] = np.array(one)
    df=pd.DataFrame(csv, columns=columns) #dataframe
    # pd.DataFrame(csv, columns=columns).to_csv('key_points.csv', index_label='Frames#') #csv

    X_test = df.loc[:,:].values

    results={}
    loaded_model = pickle.load(open('svm.pkl', 'rb'))
    y_pred=loaded_model.predict(X_test)
    unique, counts = np.unique(y_pred, return_counts=True)
    results['SVM']=unique[np.argmax(counts)]

    loaded_model = pickle.load(open('dt.pkl', 'rb'))
    y_pred=loaded_model.predict(X_test)
    unique, counts = np.unique(y_pred, return_counts=True)
    results['Decision Tree']=unique[np.argmax(counts)]

    loaded_model = pickle.load(open('gnb.pkl', 'rb'))
    y_pred=loaded_model.predict(X_test)
    unique, counts = np.unique(y_pred, return_counts=True)
    results['Gaussian NB']=unique[np.argmax(counts)]

    loaded_model = pickle.load(open('knn.pkl', 'rb'))
    y_pred=loaded_model.predict(X_test)
    unique, counts = np.unique(y_pred, return_counts=True)
    results['KNN']=unique[np.argmax(counts)]

    return results

if __name__=="__main__":
    app.run(debug=True)