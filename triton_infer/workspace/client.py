# import cv2
import numpy as np
import time
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import argparse

server_url = "192.168.106.84:8000"



def OCR(image):
    OCR_client = httpclient.InferenceServerClient(url=server_url)
    
    image_data = np.expand_dims(image, axis=0)

    inputs = []

    st= time.time()

    detection_input = httpclient.InferInput(
        "input_image", image_data.shape, datatype="UINT8"
    )

    detection_input.set_data_from_numpy(image_data)
    inputs.append(detection_input)

    # Infer
    results = OCR_client.infer(
        model_name='ensemble_model', inputs=inputs
    )

    print("Time:", time.time()-st)

    print(results.as_numpy('rec_text'))
    print('dt_boxes.shape:', results.as_numpy('dt_boxes').shape)

def OCR_grpc(image, args):
    OCR_client = grpcclient.InferenceServerClient(
        url=args.url,
        verbose=args.verbose,
    )
    
    image_data = np.expand_dims(image, axis=0)
    inputs = []
    st= time.time()

    detection_input = grpcclient.InferInput(
        "input_image", image_data.shape, datatype="UINT8"
    )
    

    detection_input.set_data_from_numpy(image_data)
    inputs.append(detection_input)

    # Infer
    results = OCR_client.infer(
        model_name=args.model_name, inputs=inputs
    )

    statistics = OCR_client.get_inference_statistics(model_name=args.model_name)
    print(statistics)
    if len(statistics.model_stats) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)

    print("Time:", time.time()-st)

    print(results.as_numpy('rec_text'))
    print('dt_boxes.shape:', results.as_numpy('dt_boxes').shape)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="ensemble_model",
        help="Model name",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    args = parser.parse_args()

    # image = cv2.imread(args.image)
    image = np.fromfile(args.image, dtype="uint8")

    OCR_grpc(image, args)