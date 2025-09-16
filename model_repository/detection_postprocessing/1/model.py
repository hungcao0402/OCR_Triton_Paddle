import triton_python_backend_utils as pb_utils
import json
import numpy as np
import cv2
import time
from det_postprocess import DBPostProcess, crop_imgs
from rec_preprocess import RecPreprocess

class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        self.det_postprocess = DBPostProcess(thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=1.5,use_dilation=False, score_mode="fast")
        self.rec_preprocessor = RecPreprocess()

        self.input_names = []
        for input in model_config['input']:
            self.input_names.append(input['name'])

        self.output_names = []
        self.output_dtype = []
        for output in model_config['output']:
            self.output_names.append(output['name'])
            dtype = pb_utils.triton_string_to_numpy(output["data_type"])
            self.output_dtype.append(dtype)

        self.rec_image_shape = (3, 48, 320)

    def execute(self, requests):
        st=time.time()
        responses = []
        for request in requests:
            det_output = pb_utils.get_input_tensor_by_name(
                request, self.input_names[0]
            )
            ori_img = pb_utils.get_input_tensor_by_name(
                request, self.input_names[1]
            )
            shape_list = pb_utils.get_input_tensor_by_name(
                request, self.input_names[2]
            )

            preds = det_output.as_numpy()
            img_raw = np.squeeze(ori_img.as_numpy(), axis=0)
            shape_list = shape_list.as_numpy()
            
            dt_boxes = self.det_postprocess(preds, shape_list)[0]['points']
            
            list_crop_img = self.rec_preprocessor.run(img_raw, dt_boxes)

            out_tensor_0 = pb_utils.Tensor(self.output_names[0], dt_boxes.astype(self.output_dtype[0])) 
            out_tensor_1 = pb_utils.Tensor(self.output_names[1], list_crop_img.astype(self.output_dtype[1])) 

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1]
            )
            
            responses.append(inference_response)
        return responses

