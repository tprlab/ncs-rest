import sys
import numpy as np
from time import time
import ncs_conf
import logging
from openvino.inference_engine import IENetwork, IEPlugin


ncs_plugin = None

def init_ncs():
    global ncs_plugin
    if ncs_plugin is not None:
        return ncs_plugin
    try:
        ncs_plugin = IEPlugin(device=ncs_conf.NCS_DEVICE, plugin_dirs=ncs_conf.NCS_PLUGIN_PATH)
        #ncs_plugin.set_config({"KEY_VPU_FORCE_RESET", "NO"})
        logging.debug("Inited NCS")
    except Exception as e:
        logging.exception("Cannot load NCS plugin")
    return ncs_plugin



def load_ncs_model(xml_path, bin_path):
    if ncs_plugin is None:
        return None, None
    t0 = time()
    model = IENetwork(model=xml_path, weights=bin_path)
    t1 = time()
    logging.debug("Read {} NCS model in {:.4f}".format(xml_path, t1 - t0))

    t0 = time()
    net = ncs_plugin.load(network=model)
    t1 = time()
    logging.debug("Loaded NCS model in {:.4f}".format(t1 - t0))

    return model, net



def run_ncs(model, net, image):
    if image is None:
        return None

    input_blob = next(iter(model.inputs))
    out_blob = next(iter(model.outputs))
    n, c, h, w = model.inputs[input_blob].shape


    images = np.ndarray(shape=(n, c, h, w))
    images[0] = image

    start = time()
    res = net.infer(inputs={input_blob: images})
    end = time()

    logging.debug("[INFO] ncs inference took {:.4f} seconds".format(end - start))

    res = res[out_blob]
    return res[0]


if __name__ == "__main__":
    ncs = init_ncs()
    device = ncs.device
    print(dir(ncs))
    #print(dir(device))