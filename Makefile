MODEL_DIR=./triton_infer/model_repository

build_container:
	docker images ppocr_onnx_exporter:latest | grep ppocr_onnx_exporter || docker compose build model_exporter
	docker images ppocr_tritonserver:latest | grep ppocr_tritonserver || docker compose build tritonserver

download_models_and_export:
	docker run -it --rm -v ./onnx_export:/onnx_export ppocr_onnx_exporter:latest bash -c "\
			cd /onnx_export && bash export.sh \
			"

copy_models:
	cp onnx_export/PP-OCRv5_mobile_det_infer/inference.onnx triton_infer/model_repository/OCRv5_det/1/OCRv5_mobile_det.onnx
	cp onnx_export/PP-OCRv5_mobile_rec_infer/inference.onnx triton_infer/model_repository/OCRv5_rec/1/OCRv5_mobile_rec.onnx
	cp onnx_export/PP-LCNet_x1_0_doc_ori_infer/inference.onnx triton_infer/model_repository/PP-LCNet_doc_ori/1/PP-LCNet_x1_0_doc_ori.onnx
	cp onnx_export/PP-LCNet_x1_0_textline_ori_infer/inference.onnx triton_infer/model_repository/PP-LCNet_textline_ori/1/PP-LCNet_textline_ori.onnx
	cp onnx_export/UVDoc_infer/inference.onnx triton_infer/model_repository/UVDoc/1/UVDoc.onnx

build_models:
	docker run -it --gpus all --shm-size=512m --rm -v $(MODEL_DIR):/models ppocr_tritonserver \
		bash -c "\
			cd /models/OCRv5_det/1 && bash trtexec.sh && \
			cd /models/OCRv5_rec/1 && bash trtexec.sh && \
			cd /models/PP-LCNet_doc_ori/1 && bash trtexec.sh && \
			cd /models/PP-LCNet_textline_ori/1 && bash trtexec.sh \
		"

setup_model_repo:
	make build_container
	make download_models_and_export
	make copy_models
	make build_models


# run_triton:
# 	docker run -it --gpus all --shm-size=512m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ./model_repository:/models nvcr.io/nvidia/tritonserver:24.07-py3
# deploy_model:
# 	tritonserver --model-repository=/models
# measure_performance:
# 	docker run -it --gpus all	\
# 		-v /var/run/docker.sock:/var/run/docker.sock \
# 		--net=host -v ${PWD}:${PWD} nvcr.io/nvidia/tritonserver:24.07-py3-sdk bash
# test_perfom_rec:
# 	perf_analyzer -m text_recognition -b 2 --shape input:3,32,320 --concurrency-range 2:16:2 --percentile=95

# docker_client:
# 	docker run -it -d --rm -v /root/workspace/fullflow:/root/workspace/fullflow pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel bash
# install:
# 	apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# test:
# 	model-analyzer profile \
# 		--model-repository /mnt/01DAF375F466EC40/hungcv/paddleocr_onnx_triton/triton_infer/model_repository	\
# 		--profile-models text_recognition 	\
# 		--output-model-repository-path /mnt/01DAF375F466EC40/hungcv/paddleocr_onnx_triton/triton_infer/model_repository/output/  \
# 		--override-output-model-repository --latency-budget 100 \
# 		--run-config-search-mode quick \
# 		--triton-launch-mode=docker
