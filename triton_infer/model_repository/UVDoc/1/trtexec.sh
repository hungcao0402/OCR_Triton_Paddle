/usr/src/tensorrt/bin/trtexec  --onnx=UVDoc.onnx \
  --minShapes=image:1x3x128x128 \
  --optShapes=image:1x3x512x512 \
  --maxShapes=image:8x3x1024x1024 \
  --saveEngine=uvdoc.plan \
  --timingCacheFile=timing_cache.json \
  --fp16
