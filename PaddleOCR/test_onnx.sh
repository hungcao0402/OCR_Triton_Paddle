python tools/infer/predict_system.py --use_gpu=True --use_onnx=True \
    --det_model_dir=./inference/det_onnx/model.onnx  \
    --rec_model_dir=./inference/rec_onnx/model.onnx  \
    --image_dir=img_12.jpg \
    --rec_char_dict_path=ppocr/utils/en_dict.txt

python  tools/infer/predict_system.py --use_gpu=True \
    --rec_model_dir=./inference/en_PP-OCRv3_rec_infer \
    --det_model_dir=./inference/en_PP-OCRv3_det_infer \
    --image_dir=./img_12.jpg \
    --rec_char_dict_path=ppocr/utils/en_dict.txt
