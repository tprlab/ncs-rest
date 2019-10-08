import json
import pytest
import time
import unittest
import ncs_client
import numpy as np


class ClientTest(unittest.TestCase):

    def test_ssd_inference_file(self):
        rc, out = ncs_client.run_inference_file("mobile_ssd", "data/pic.jpg")
        self.assertTrue(rc)
        self.assertEqual(out.shape, (1, 100, 7))


    def test_ssd_inference_path(self):
        rc, out = ncs_client.run_inference_path("mobile_ssd", "data/pic.jpg")
        self.assertTrue(rc)
        self.assertEqual(out.shape, (1, 100, 7))

    def test_inception_inference_path(self):
        rc, out = ncs_client.run_inference_path("inception_v4_inference_graph", "data/pic.jpg")
        self.assertTrue(rc)
        self.assertEqual(out.shape[0], 1001)

    def test_seg_inference_path(self):
        rc, out = ncs_client.run_inference_path("road-segmentation-adas-0001", "data/road.jpg")
        self.assertTrue(rc)
        self.assertEqual(out.shape, (4, 512, 896))

    def test_detection_path(self):
        rc, out = ncs_client.detect_path("mobile_ssd", "data/pic.jpg")
        self.assertTrue(rc)
        rs = {}
        for e in out:
            rs[e["class"]] = e

        self.assertEqual(rs[1]["score"], 51)
        self.assertEqual(rs[18]["score"], 52)
        self.assertEqual(rs[85]["score"], 51)
        self.assertEqual(rs[17]["score"], 59)

    def test_inception_file(self):
        rc, out = ncs_client.classify_file("inception_v4_inference_graph", "data/pic.jpg")
        self.assertTrue(rc)
        rs = {}
        for e in out:
            rs[e["class"]] = e["score"]

        self.assertEqual(rs[621], 85)
        self.assertEqual(rs[682], 8)

    def test_seg_file(self):
        rc, out = ncs_client.segment_file("road-segmentation-adas-0001", "data/road.jpg")
        self.assertTrue(rc)
        self.assertEqual(out.shape, (240, 320))
        unique, counts = np.unique(out, return_counts=True)
        d = dict(zip(unique, counts))
        self.assertEqual(d[0], 45892)
        self.assertEqual(d[1], 30893)
        self.assertEqual(d[2], 15)

    def test_seg_shape(self):
        sh = ncs_client.get_input_shape("road-segmentation-adas-0001")
        self.assertEqual(sh, [1, 3, 512, 896])

    def test_detect_shape(self):
        sh = ncs_client.get_input_shape("mobile_ssd")
        self.assertEqual(sh, [1, 3, 300, 300])

    def test_class_shape(self):
        sh = ncs_client.get_input_shape("inception_v4_inference_graph")
        self.assertEqual(sh, [1, 3, 299, 299])


