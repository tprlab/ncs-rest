LOG_PATH="/home/pi/ncsapi/logs"
LOG_FILE="ncs_api.log"

NCS_DEVICE = "MYRIAD"
NCS_PLUGIN_PATH = "/opt/intel/openvino/inference_engine/lib/armv7l"

MODELS_PATH = "/home/pi/ncs_models"
CLASS_MODEL = "inception_v4_inference_graph"
DETECT_MODEL = "person-detection-retail-0002-fp16"
SEGMENT_MODEL = "road-segmentation-adas-0001"
MOBILE_SSD = "mobile_ssd"

#LOAD_ON_START = [MOBILE_SSD, CLASS_MODEL, SEGMENT_MODEL]
LOAD_ON_START = [MOBILE_SSD]