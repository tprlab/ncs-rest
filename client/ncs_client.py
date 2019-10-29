import requests
import time
import json
import sys
import os
import numpy as np
import io
import cv2 as cv


NCS_URL = "http://localhost:5001"

colors = [
    (0, 0, 0),
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0)
]

colors = np.array(colors, dtype="uint8")


def set_url(url):
    NCS_URL = url

def get_url():
    return NCS_URL

def read_np_array(rsp):
    memfile = io.BytesIO()
    memfile.write(rsp.content)
    memfile.seek(0)
    return np.load(memfile)

def read_memfile(path):
    with open(path, "rb") as f:
        content = f.read()
        memfile = io.BytesIO()
        memfile.write(content)
        memfile.seek(0)
        return memfile



def read_json(rsp):
    return rsp.json()


def get_models():
    t0 = time.time()
    rsp = requests.get(NCS_URL + "/list")
    t1 = time.time()
    print("Request done in {:.4f}".format(t1 - t0))
    if rsp.status_code == requests.codes.ok:
        return True, rsp.json()
    return False, rsp.content

def request_with_file(url, path, trans_proc):
    with open(path, "rb") as f:
        params = dict (file = f)
        rsp = requests.post(url, files=params, verify=False)
        if rsp.status_code == requests.codes.ok:
            return True, rsp if trans_proc is None else trans_proc(rsp)
        return False, rsp.content

def request_with_memfile(url, f, trans_proc):
    params = dict (file = f)
    rsp = requests.post(url, files=params, verify=False)
    if rsp.status_code == requests.codes.ok:
        return True, rsp if trans_proc is None else trans_proc(rsp)
    return False, rsp.content


def request_with_path(url, path, trans_proc):
    headers = {'Content-type': 'application/json'}
    path = os.path.abspath(path)
    rsp = requests.post(url, headers=headers, data=json.dumps({"path" : path}), verify=False)
    if rsp.status_code == requests.codes.ok:
        return True, rsp if trans_proc is None else trans_proc(rsp)
    return False, rsp.content



def run_inference_file(model, path):
    return request_with_file(NCS_URL + "/inference/file/" + model, path, read_np_array)

def run_inference_path(model, path):
    return request_with_path(NCS_URL + "/inference/path/" + model, path, read_np_array)

def classify_file(model, path):
    return request_with_file(NCS_URL + "/classify/file/" + model, path, read_json)

def classify_path(model, path):
    return request_with_path(NCS_URL + "/classify/path/" + model, path, read_json)

def detect_file(model, path):
    return request_with_path(NCS_URL + "/detect/file/" + model, path, read_json)

def detect_path(model, path):
    return request_with_path(NCS_URL + "/detect/path/" + model, path, read_json)

def segment_file(model, path):
    return request_with_file(NCS_URL + "/segment/file/" + model, path, read_np_array)

def segment_path(model, path):
    return request_with_path(NCS_URL + "/segment/path/" + model, path, read_np_array)

def get_input_shape(model):
    rsp = requests.get(NCS_URL + "/input/shape/{}".format(model))
    if rsp.status_code != requests.codes.ok:
        return None
    return rsp.json()
        

def test_segment(model, path):
    #rc, out = segment(model, path)
    rc, out = segment_path(model, path)
    if not rc:
        print("Seg failed", rc, out)
        return rc, out
    print(out.shape)
    unique, counts = np.unique(out, return_counts=True)
    d = dict(zip(unique, counts))
    print(d)

    mask = colors[out]
    img = cv.imread(path)
    output = ((0.6 * img) + (0.4 * mask)).astype("uint8")
    cv.imwrite("seg_test.jpg", output);


if __name__ == "__main__":
    """
    rc, data = get_models()
    if rc:
        for l in data:
            print(l)
    else:
        print(data)
    """

    #rc, out = run_inference_file("mobile_ssd", "data/pic.jpg")
    #rc, out = run_inference_file("inception_v4_inference_graph", "data/pic.jpg")
    #rc, out = run_inference_file("road-segmentation-adas-0001", "data/road.jpg")
    #rc, out = run_inference_path("mobile_ssd", "data/pic.jpg")
    rc, out = classify_path("inception_v4_inference_graph", "data/pic.jpg")
    #rc, out = detect("mobile_ssd", "data/pic.jpg")
    
    #rc, out = detect_path("mobile_ssd", "data/pic.jpg")

    if rc:
        print("Inference", out)
        #print(out.shape)
    else:
        print("Inference error:", out)

    
    #test_segment("road-segmentation-adas-0001", "data/road.jpg")
    #rs = get_input_shape("road-segmentation-adas-0001")
    #rs = get_input_shape("inception_v4_inference_graph")
    #rs = get_input_shape("mobile_ssd")
    #print(rs)

    



 