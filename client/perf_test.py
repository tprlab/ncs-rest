import requests
import time
import json
import sys
import os
import numpy as np
import ncs_client

path = "seg_test.jpg"

mf = ncs_client.read_memfile(path)

t0 = time.time()
for i in range(0, 1000):
    rc, rs = ncs_client.request_with_memfile(ncs_client.NCS_URL + "/test/file", mf, None)
    mf.seek(0)
t1 = time.time()

print("Test with file was done with {:.4f} seconds".format(t1 - t0))


t0 = time.time()
for i in range(0, 1000):
    rc, rs = ncs_client.request_with_path(ncs_client.NCS_URL + "/test/path", path, None)
t1 = time.time()

print("Test with path was done with {:.4f} seconds".format(t1 - t0))
