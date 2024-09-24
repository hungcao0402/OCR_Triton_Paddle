import time
import triton_python_backend_utils as pb_utils
import json
import numpy as np
from rec_postprocess import CTCLabelDecode
import os
    
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

        # charracter_dict=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/']
        dir_name = os.path.dirname(os.path.realpath(__file__)) 
        dict_path = os.path.join(dir_name, 'en_dict.txt')

        self.postprocess_op = CTCLabelDecode(dict_path)
        
    def execute(self, requests):
        st = time.time()
        responses = []
        for request in requests:
            in_1 = pb_utils.get_input_tensor_by_name(request, self.input_names[0])
            in_1 = in_1.as_numpy()

            results= self.postprocess_op(in_1)
            out_tensor_0 = pb_utils.Tensor(self.output_names[0], np.array(results[0], dtype=np.object_).astype(self.output_dtype[0]))
            out_tensor_1 = pb_utils.Tensor(self.output_names[1], np.array(results[1]).astype(self.output_dtype[1]))

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, out_tensor_1])
            
            responses.append(inference_response)
        print('rec post:', time.time()-st)
        return responses
