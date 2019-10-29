import ncs_core
import ncs_wrapper
import logging
import model_loader
import ncs_conf
import cv2 as cv
import numpy as np

ncs_models = model_loader.get_models()


def test_file(data):
    return decode_image_data(data)


def test_path(path):
    rc, img = load_image(path)
    return img


def init_models():
    if len(ncs_conf.LOAD_ON_START) > 0:
        for m in ncs_conf.LOAD_ON_START:
            logging.debug(("Load-on-start", m))
            load_model(ncs_conf.MODELS_PATH, m)


def list_models():
    return list(ncs_models.keys())

def load_model(path, name):
    return model_loader.load_model(path, name)

def unload_model(name):
    model_loader.unload_model(name)



def get_input_shape(model):
    if not model in ncs_models:
         return None
    return ncs_models[model].get_input_shape()

def run_inference(model_id, img):
    if not model_id in ncs_models:
        return False, "Model " + model_id + " not found"
    model = ncs_models[model_id]
    ret = model.run(img, True)
    if ret is None:
        return False, "No inference output"
    return True, ret

def decode_image_data(data):
    return cv.imdecode(data, cv.IMREAD_UNCHANGED)

def load_image(path):
    img = cv.imread(path)
    if img is None:
        return False, "File {} not found".format(path)
    return True, img

def run_inference_file(model_id, data):
    img = decode_image_data(data)
    return run_inference(model_id, img)

def run_inference_path(model_id, path):
    rc, img = load_image(path)
    if not rc:
        return rc, img
    return run_inference(model_id, img)

def get_class_tensor(data):
    ret = []
    thr = 0.01
    while(True):
        cls = np.argmax(data)
        if data[cls] < thr:
            break;
        logging.debug(("Class", cls, "score", data[cls]))
        c = {"class" : int(cls), "score" : int(100 * data[cls])}
        data[cls] = 0
        ret.append(c)
    return ret

def classify(model_id, img):
    rc, out = run_inference(model_id, img)
    if not rc:
        return rc, out
    return True, get_class_tensor(out)


def classify_file(model_id, data):
    img = decode_image_data(data)
    return classify(model_id, img)

def classify_path(model_id, path):
    rc, img = load_image(path)
    if not rc:
        return rc, img
    return classify(model_id, img)

def get_detect_from_tensor(t, rows, cols):
    score = int(100 * t[2])
    cls = int(t[1])
    left = int(t[3] * cols)
    top = int(t[4] * rows)
    right = int(t[5] * cols)
    bottom = int(t[6] * rows)

    return {"class" : cls, "score" : score, "x" : left, "y" : top, "w" : (right - left), "h" : (bottom - top)}


def build_detection(data, thr, rows, cols):
    T = {}
    for t in data:
        score = t[2]
        if score > thr:
            cls = int(t[1])
            if cls not in T:
                T[cls] = get_detect_from_tensor(t, rows, cols)
            else:
                a = T[cls]
                if a["score"] < score:
                    T[cls] = get_detect_from_tensor(t, rows, cols)
    return list(T.values())


def detect(model_id, img):
    rc, out = run_inference(model_id, img)
    if not rc:
        return rc, out

    rows, cols = img.shape[:2]
    return True, build_detection(out[0], 0.01, rows, cols)

def detect_file(model_id, data):
    img = decode_image_data(data)
    return detect(model_id, img)

def detect_path(model_id, path):
    rc, img = load_image(path)
    if not rc:
        return rc, img

    return detect(model_id, img)

def segment(model_id, img):
    rc, out = run_inference(model_id, img)
    if not rc:
        return rc, out

    out = np.argmax(out, axis=0)
    out = cv.resize(out, (img.shape[1], img.shape[0]),interpolation=cv.INTER_NEAREST)
    return True, out

def segment_file(model_id, data):
    img = decode_image_data(data)
    return segment(model_id, img)

def segment_path(model_id, path):
    rc, img = load_image(path)
    if not rc:
        return rc, img

    return segment(model_id, img)
