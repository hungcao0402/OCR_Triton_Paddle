/usr/src/tensorrt/bin/trtexec --onnx=OCRv5_mobile_rec.onnx \
  --minShapes=x:1x3x48x160 \
  --optShapes=x:8x3x48x320 \
  --maxShapes=x:64x3x48x1280 \
  --saveEngine=rec.plan \
  --timingCacheFile=timing_cache.json \
  --fp16