FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    git wget unzip libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install paddlepaddle==3.2.0 paddlex==3.2.1 onnx==1.17.0 onnxruntime==1.22.1
RUN paddlex --install paddle2onnx
CMD ["bash"]
