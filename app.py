import os
import threading
import datetime
import time
import requests
import logging
import io
import numpy as np

import ncs_conf as conf

if not os.path.isdir(conf.LOG_PATH):
    os.makedirs(conf.LOG_PATH)        

log_file = conf.LOG_PATH + "/" + conf.LOG_FILE
logging.basicConfig(filename=log_file,level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(threadName)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


from flask import Flask
from flask import send_file, send_from_directory
from flask import jsonify
from flask import request


import ncs_ctrl


app = Flask(__name__)

ncs_ctrl.init_models()

def send_np_array(a):
    out_file = io.BytesIO()
    np.save(out_file, a)
    out_file.seek(0)
    return send_file(out_file, mimetype="application/octet-stream")



@app.route('/')
def index():
    return 'NCS REST Service'


@app.route('/list', methods=['GET'])
def list_models():
    return jsonify(ncs_ctrl.list_models()), requests.codes.ok

@app.route('/input/shape/<model>', methods=['GET'])
def get_model_vars(model):
    ret = ncs_ctrl.get_input_shape(model)
    if ret is None:
        return "model not found", requests.codes.bad_request

    return jsonify(ret), requests.codes.ok



@app.route('/load', methods=['POST'])
def load_model():
    content = request.get_json()
    if not "path" in content:
        return jsonify({"missed" : "path"}), requests.codes.bad_request

    if not "name" in content:
        return jsonify({"missed" : "name"}), requests.codes.bad_request


    path = content["path"]
    name = content["name"]

    rc, err = ncs_ctrl.load_model(path, name)
    if rc:
        return jsonify({}), requests.codes.created 
    return jsonify({"error" : err}), requests.codes.unavailable


@app.route('/unload/<model>', methods=['POST'])
def unload_model(model):
    ncs_ctrl.unload_model(model)
    return model, requests.codes.ok


def get_request_file(request):
    if 'file' not in request.files:
        return None

    file = request.files['file']
    input_file = io.BytesIO()
    file.save(input_file)
    return np.fromstring(input_file.getvalue(), dtype=np.uint8)


@app.route('/inference/file/<model>', methods=['POST'])
def run_inference(model):
    data = get_request_file(request)
    if data is None:
        "file", requests.codes.bad_request
    logging.debug(("Got data for inference", len(data)))
    rc, ret = ncs_ctrl.run_inference_file(model, data)
    if not rc:
        return jsonify({"error" : ret}), requests.codes.bad_request
    logging.debug(("Sending inference", ret.shape))
    return send_np_array(ret)

@app.route('/inference/path/<model>', methods=['POST'])
def run_inference_file(model):
    content = request.get_json()
    if not "path" in content:
        return jsonify({"missed" : "path"}), requests.codes.bad_request

    path = content["path"]
    logging.debug(("Run inference for ", model, "from file ", path))
    rc, ret = ncs_ctrl.run_inference_path(model, path)
    if not rc:
        return jsonify({"error" : ret}), requests.codes.bad_request
    logging.debug(("Sending inference", ret.shape))
    return send_np_array(ret)

@app.route('/classify/file/<model>', methods=['POST'])
def classify(model):
    data = get_request_file(request)
    if data is None:
        "file", requests.codes.bad_request

    rc, ret = ncs_ctrl.classify_file(model, data)
    if not rc:
        return jsonify({"error" : ret}), requests.codes.bad_request
    return jsonify(ret), requests.codes.ok


@app.route('/classify/path/<model>', methods=['POST'])
def classify_file(model):
    content = request.get_json()
    if not "path" in content:
        return jsonify({"missed" : "path"}), requests.codes.bad_request

    path = content["path"]
    rc, ret = ncs_ctrl.classify_path(model, path)
    if not rc:
        return jsonify({"error" : ret}), requests.codes.bad_request
    return jsonify(ret), requests.codes.ok


@app.route('/detect/file/<model>', methods=['POST'])
def detect(model):
    data = get_request_file(request)
    if data is None:
        "file", requests.codes.bad_request

    rc, ret = ncs_ctrl.detect_file(model, data)
    if not rc:
        return jsonify({"error" : ret}), requests.codes.bad_request
    return jsonify(ret), requests.codes.ok


@app.route('/detect/path/<model>', methods=['POST'])
def detect_file(model):
    content = request.get_json()
    if not "path" in content:
        return jsonify({"missed" : "path"}), requests.codes.bad_request

    path = content["path"]

    rc, ret = ncs_ctrl.detect_path(model, path)
    if not rc:
        return jsonify({"error" : ret}), requests.codes.bad_request
    return jsonify(ret), requests.codes.ok

@app.route('/segment/file/<model>', methods=['POST'])
def segment(model):
    data = get_request_file(request)
    if data is None:
        "file", requests.codes.bad_request

    rc, ret = ncs_ctrl.segment_file(model, data)
    if not rc:
        return jsonify({"error" : ret}), requests.codes.bad_request
    return send_np_array(ret)

@app.route('/segment/path/<model>', methods=['POST'])
def segment_file(model):
    content = request.get_json()
    if not "path" in content:
        return jsonify({"missed" : "path"}), requests.codes.bad_request

    path = content["path"]

    rc, ret = ncs_ctrl.segment_path(model, path)
    if not rc:
        return jsonify({"error" : ret}), requests.codes.bad_request
    return send_np_array(ret)





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True, use_reloader=False)

