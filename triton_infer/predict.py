from triton_utils.client.triton_client import TritonClient, force_free_shm
import cv2

force_free_shm('localhost:8001')
client = TritonClient('localhost:8001', 'det-rec-ensemble', shm=False)

import glob
crops = glob.glob('ocr_crops/*.png')
for crop in sorted(crops):
    img = cv2.imread(crop)
    print(crop, img.shape)
    try:
        result = client.triton_predict(0, [img[None]])
        for txt, conf in zip(result['rec_text'], result['rec_score']):
            print(f'text: {txt.decode("utf-8")} ({conf:.2f})')
    except:
        pass