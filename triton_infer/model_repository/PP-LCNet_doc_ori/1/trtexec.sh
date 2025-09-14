/usr/src/tensorrt/bin/trtexec  --onnx=PP-LCNet_x1_0_doc_ori.onnx \
  --minShapes=x:1x3x224x224 \
  --optShapes=x:1x3x224x224 \
  --maxShapes=x:8x3x224x224 \
  --saveEngine=doc_ori.plan \
  --timingCacheFile=timing_cache.json \
  --fp16
