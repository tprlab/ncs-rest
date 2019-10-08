import json
import pytest
import time
import unittest
import ncs_conf

from app import app

class NcsTest(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client

    def test_load_no_path(self):
        name = ncs_conf.SEGMENT_MODEL
        headers = {'Content-type': 'application/json'}
        request = self.client().post("/load", headers=headers, data=json.dumps({"name" : name}))
        self.assertEqual(request.status_code, 400)

    def test_load_no_name(self):
        headers = {'Content-type': 'application/json'}
        request = self.client().post("/load", headers=headers, data=json.dumps({"path" : ncs_conf.MODELS_PATH}))
        self.assertEqual(request.status_code, 400)

    def test_load_no_ctype(self):
        name = ncs_conf.SEGMENT_MODEL
        request = self.client().post("/load", data=json.dumps({"path" : ncs_conf.MODELS_PATH, "name" : name}))
        self.assertEqual(request.status_code, 500)

    def load_model(self, name):
        headers = {'Content-type': 'application/json'}
        return self.client().post("/load", headers=headers, data=json.dumps({"path" : ncs_conf.MODELS_PATH, "name" : name}))

    def unload_model(self, name):
        return self.client().post("/unload/{}".format(name))

    def load_model_and_check(self, name, remove = False):
        resp = self.load_model(name)
        self.assertEqual(resp.status_code, 201)

        lst = self.get_list()
        self.assertTrue(len(lst) > 0)
        self.assertTrue(name in lst)

        sh = self.get_input_shape(name);
        self.assertTrue(len(sh) > 0)

        if remove:
            resp = self.unload_model(name)
            self.assertEqual(resp.status_code, 200)



    def test_load_road(self):
        self.load_model_and_check(ncs_conf.SEGMENT_MODEL, True)
        
    """
    def test_load_mult(self):
        self.test_load_road()
        self.test_load_detect()
        self.test_load_classify()
    """

    def test_load_nf(self):
        request = self.load_model("absent")
        self.assertEqual(request.status_code, 503)

    def test_load_detect(self):
        self.load_model_and_check(ncs_conf.DETECT_MODEL, True)

    def test_load_classify(self):
        self.load_model_and_check(ncs_conf.CLASS_MODEL, True)

    def get_list(self):
        resp = self.client().get("/list")
        self.assertEqual(resp.status_code, 200)
        return resp.get_json()

    def test_empty_list(self):
        l = self.get_list()
        self.assertEqual(len(l), 0)


    def get_input_shape(self, model):
        resp = self.client().get("/input/shape/{}".format(model))
        self.assertEqual(resp.status_code, 200)
        return resp.get_json()

    def check_shape(self, model, shape):
        self.load_model_and_check(model)
        rs = self.get_input_shape(model)
        self.assertEqual(rs, shape)
        self.unload_model(model)

    def test_cls_shape(self):
        self.check_shape(ncs_conf.CLASS_MODEL, [1, 3, 299, 299])

    def test_detect_shape(self):
        self.check_shape(ncs_conf.MOBILE_SSD, [1, 3, 300, 300])

    def test_seg_shape(self):
        self.check_shape(ncs_conf.SEGMENT_MODEL, [1, 3, 512, 896])












