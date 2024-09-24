import triton_python_backend_utils as pb_utils
import json
import sys
import cv2
import time
from det_preprocess import DetPreprocess
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])


        self.input_names = []
        for input in model_config['input']:
            self.input_names.append(input['name'])

        self.output_names = []
        self.output_dtype = []
        for output in model_config['output']:
            self.output_names.append(output['name'])
            dtype = pb_utils.triton_string_to_numpy(output["data_type"])
            self.output_dtype.append(dtype)

        self.det_preprocess = DetPreprocess()

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.
        Every Python backend must iterate through list of requests and 
        create an instance of pb_utils.InferenceResponse class for each
        of them. Reusing the same pb_utils.InferenceResponse object for 
        multiple requests may result in segmentation faults. 
        You should avoid storing any of the input Tensors in the class
        attributes as they will be overridden in subsequent inference requests.
        You can make a copy of the underlying NumPy array and store it if it is required.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        st= time.time()
        responses = []
        for request in requests:
            imgs = pb_utils.get_input_tensor_by_name(request, self.input_names[0])
            imgs = imgs.as_numpy()

            st_pr = time.time()
            outputs, im_infos = self.det_preprocess.run(imgs)

            output_tensor_0 = pb_utils.Tensor(self.output_names[0], outputs[0].astype(self.output_dtype[0])) 
            output_tensor_1 = pb_utils.Tensor(self.output_names[1], im_infos[0].astype(self.output_dtype[1])) # output 1 image

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor_0, output_tensor_1])
            responses.append(inference_response)

        print('det pre:', time.time()-st)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
