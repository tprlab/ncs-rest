# NCS-REST

This is REST wrapper based on flask for [Intel NCS](https://software.intel.com/en-us/articles/run-intel-openvino-models-on-intel-neural-compute-stick-2).

This is a great hardware to accelerate neural networks inferences but it does not support an [access from multiple processes](https://docs.openvinotoolkit.org/2019_R1.1/_docs_IE_DG_supported_plugins_MYRIAD.html#supported_configuration_parameters").

So far I needed exactly this scenario I made this wrapper with REST interface:
(Also there is ncs_client python module to work with the service without the protocol details).

**POST: /load**

  *Loads a model on NCS*
  
  Input: JSON 
  
    * Parameters:
    
      * name - name of the model (xml and bin files must have the same name with different extension)
      
      * path - path to the model files
      
  Output:
  
    * 201 if success
    
    * 503 otherwise
    
 **POST: /unload/$model**
  
   *Removes the model from the service (Looks like NCS does not support unload of the models)*
   
   Output: 
   
     * 200
 
 **GET: /list**
 
   *Lists all models loaded on NCS and available via the sevice*
   
   Output:
   
     * 200
     
     * List of loaded models in JSON format
     
 **GET: /input/shape/$model** 
  
   *Returns shape of an input tensor of the specified model*
   
   Output:
   
     * 200
     
     * JSON array with dimensions 

  **POST: /inference/file/$model**
  
    *Makes an inference from the specified model on input data.*
  
    Input:
    
      * Binary content of the image file passed as multipart
    
    Output:
    
      * Output tensor represented as serialized numpy array. Refer ncs_client for the details.
 
   **POST: /inference/path/$model**
  
     *Makes an inference from the specified model on input data.*
  
     Input:
     
       * Path to image. Assumed the image is available via filesystem.
    
     Output:
     
       * Output tensor represented as serialized numpy array. Refer ncs_client for the details.
 
 **POST: /classify/file/$model**
  
  *Does image classification with the specified model*
  
  Input:
  
    * Same as /inference/file/$model
  
  Output:
  
    * JSON array with classes and probabilities
 
 **POST: /classify/path/$model**
  
  *Does image classification with the specified model*
  
  Input:
  
    * Same as /inference/path/$model
    
  Output:
  
    * JSON array with classes and probabilities
 
 **POST: /detect/file/$model**
  
  *Does object detection on input image with the specified model*
  
  Input:
  
    * Same as /inference/file/$model
  
  Output:
  
    * JSON array with classes, coordinates and probabilities
 
 **POST: /detect/path/$model**
  
  *Does object detection on input image with the specified model*
  
  Input:
  
    * Same as /inference/path/$model
  
  Output:
  
    * JSON array with classes, coordinates and probabilities
 
 **POST: /segment/file/$model**
  
  *Does semantic segmentation of an input image with the specified model*
  
  Input:
  
    * Same as /inference/file/$model
  
  Output:
  
    * Matrix with most probable classes for each pixel as serialized numpy array
 
 **POST: /segment/path/$model**
  
  *Does semantic segmentation of an input image with the specified model*
  
  Input:
  
    * Same as /inference/path/$model
  
  Output:
  
    * Matrix with most probable classes for each pixel as serialized numpy array
