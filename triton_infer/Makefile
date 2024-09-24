run_triton:
	docker run -it --gpus all --shm-size=512m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ./model_repository:/models nvcr.io/nvidia/tritonserver:24.07-py3
deploy_model:
	tritonserver --model-repository=/models
measure_performance:
	docker run -it --gpus all	\
		-v /var/run/docker.sock:/var/run/docker.sock \
		--net=host -v ${PWD}:${PWD} nvcr.io/nvidia/tritonserver:24.07-py3-sdk bash
test_perfom_rec:
	perf_analyzer -m text_recognition -b 2 --shape input:3,32,320 --concurrency-range 2:16:2 --percentile=95

docker_client:
	docker run -it -d --rm -v /root/workspace/fullflow:/root/workspace/fullflow pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel bash
install:
	apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

test:
	model-analyzer profile \
		--model-repository /mnt/01DAF375F466EC40/hungcv/paddleocr_onnx_triton/triton_infer/model_repository	\
		--profile-models text_recognition 	\
		--output-model-repository-path /mnt/01DAF375F466EC40/hungcv/paddleocr_onnx_triton/triton_infer/model_repository/output/  \
		--override-output-model-repository --latency-budget 100 \
		--run-config-search-mode quick \
		--triton-launch-mode=docker
