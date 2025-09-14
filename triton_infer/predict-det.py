import numpy as np
import os, glob
from triton_utils.client.triton_client import TritonClient, force_free_shm
import cv2

force_free_shm('localhost:8001')
client = TritonClient('localhost:8001', 'det-rec-ensemble', shm=False)
# client = TritonClient('localhost:8001', 'det-ensemble', shm=False)
# client2 = TritonClient('localhost:8001', 'det-post-ensemble', shm=False, worker_name='post')

# from paddlex.inference.models.text_detection.processors import DBPostProcess, DetResizeForTest
# postprocess = DBPostProcess()

texts = list()

crops = glob.glob('ocr_crops/*.png')
for i, crop in enumerate(sorted(crops)):
    img = cv2.imread(crop)
    print(crop, img.shape)
    
    result = client.triton_predict(0, [img[None]])
    for r in result['rec_text']:
        text = r.decode('utf-8')
        if text.startswith('N') and len(text) == 10:
            print(text)
            texts.append(text)
    # heatmap = (result['detections']*255)[0,0]
    # os.makedirs('out_hm', exist_ok=True)
    # cv2.imwrite(f'out_hm/{i}.jpg', np.hstack((cv2.resize(img, (512, 512))[...,0], heatmap)))
    
    # prc = postprocess([result['detections']], result['shape_list'])

    # for box, score in zip(prc[0][0], prc[1][0]):
    #     cv2.polylines(img, [box.astype('int32')], True, (255,0,0), 1, cv2.LINE_4)
    # os.makedirs('out_box', exist_ok=True)
    # cv2.imwrite(f'out_box/{i}.jpg', img)

    # client2.triton_predict(0, [img[None]])

print(texts)
print(len(texts))
print(set(texts))
print(len(set(texts)))