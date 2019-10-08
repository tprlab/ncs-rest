# ncs-rest

This is REST wrapper based on flask for [Intel NCS](https://software.intel.com/en-us/articles/run-intel-openvino-models-on-intel-neural-compute-stick-2)

This is a great hardware to accelerate neural networks inferences but it does not support an [access from multiple processes](https://docs.openvinotoolkit.org/2019_R1.1/_docs_IE_DG_supported_plugins_MYRIAD.html#supported_configuration_parameters").

So far I needed exactly this scenario I made this wrapper with REST interface:
* /load (POST)
  * Input: json 
  * Parameters:
    * name - name of the model (must be equal for both xml and bin files)
    * path - path to the model files
