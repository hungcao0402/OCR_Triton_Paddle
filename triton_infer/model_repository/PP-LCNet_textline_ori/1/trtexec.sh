/usr/src/tensorrt/bin/trtexec  --onnx=PP-LCNet_x1_0_textline_ori.onnx \
  --minShapes=x:1x3x80x160 \
  --optShapes=x:1x3x80x160 \
  --maxShapes=x:8x3x80x160 \
  --saveEngine=textline_ori.plan \
  --timingCacheFile=timing_cache.json \
  --fp16
