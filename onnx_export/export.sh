#!/bin/bash
set -e

download_and_convert () {
    MODEL_NAME=$1
    URL=$2

    if [ -d "$MODEL_NAME" ] && ls "$MODEL_NAME"/*.onnx >/dev/null 2>&1; then
        echo "[✔] $MODEL_NAME already has ONNX file, skipping..."
    else
        if [ ! -d "$MODEL_NAME" ]; then
            echo "[↓] Downloading and extracting $MODEL_NAME ..."
            wget -qO- "$URL" | tar -xvf -
        else
            echo "[!] $MODEL_NAME exists but no ONNX file found, converting..."
        fi

        echo "[⚙] Converting $MODEL_NAME to ONNX ..."
        paddlex --paddle2onnx \
            --paddle_model_dir "$MODEL_NAME/" \
            --onnx_model_dir "$MODEL_NAME/" \
            --opset_version 17
    fi
}

# Detection
download_and_convert "PP-OCRv5_mobile_det_infer"  "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar"
download_and_convert "PP-OCRv5_server_det_infer"  "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar"

# Recognition
download_and_convert "PP-OCRv5_mobile_rec_infer"  "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar"
download_and_convert "PP-OCRv5_server_rec_infer"  "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar"

# Textline orientation
download_and_convert "PP-LCNet_x0_25_textline_ori_infer" "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar"
download_and_convert "PP-LCNet_x1_0_textline_ori_infer"  "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar"

# Doc orientation
download_and_convert "PP-LCNet_x1_0_doc_ori_infer" "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar"

# Text unwarping
download_and_convert "UVDoc_infer" "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar"
