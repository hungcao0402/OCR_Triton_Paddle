/usr/src/tensorrt/bin/trtexec  --onnx=OCRv5_mobile_det.onnx \
  --minShapes=x:1x3x32x328 \
  --optShapes=x:1x3x512x512 \
  --maxShapes=x:1x3x1024x1024 \
  --saveEngine=det.plan \
  --timingCacheFile=timing_cache.json \
  --fp16
