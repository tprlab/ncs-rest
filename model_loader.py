import ncs_core
import ncs_wrapper
import logging
import threading
import queue
import time

ncs_models = {}
load_queue = queue.Queue();

def get_models():
    return ncs_models


class NcsModel(ncs_wrapper.NcsWrapper):
    
    def __init__(self, name):
        self.name = name


def load_proc():
    logging.debug("Load thread started")
    while(True):
        elem = load_queue.get()
        path = elem["path"]
        name = elem["name"]

        xml_path = path + "/" + name + ".xml"
        bin_path = path + "/" + name + ".bin"

        cond = elem["condition"]
        logging.debug("Read from load Q: {}".format(xml_path))

        try:
            ncs_model = NcsModel(name)
            ncs_model.load_model_base(xml_path, bin_path)
            ncs_models[name] = ncs_model

        except Exception as e:
            logging.exception("Cannot load ncs model")
            elem["error"] = str(e)

        with cond:
            cond.notify()


load_thread = threading.Thread(target=load_proc)
load_thread.setDaemon(True)
load_thread.start()


def load_model(path, name):
    logging.debug(("Started model loading", name))
    if name in ncs_models:
        return True, None
        
    cond = threading.Condition()
    e = {"path" : path, "name" : name, "condition" : cond}

    load_queue.put(e)
    with cond:
        cond.wait()
    ok = not "error" in e
    logging.debug(("Finished model loading", name, ok))
    
    if not ok:
        return False, e["error"]
    return True, None

def unload_model(name):
    logging.debug(("Unloading model", name))
    if name in ncs_models:
        del ncs_models[name]

if __name__ == "__main__":
    import ncs_conf
    t0 = time.time()
    rc, err = load_model(ncs_conf.MODELS_PATH, "absent")
    #ncs_conf.SEGMENT_MODEL)
    t1 = time.time()
    print("Model {} loaded in {:.4f} sec".format(ncs_conf.SEGMENT_MODEL, (t1 - t0)))

