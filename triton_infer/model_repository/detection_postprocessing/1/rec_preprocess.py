import numpy as np
import cv2
import math

class RecPreprocess:
    rec_image_shape = ( 3, 48, 320)

    def run(self, img_raw, dt_boxes):
        crop_coords = [crop.astype(int) for crop in dt_boxes]
        list_crop_img = self.crop_imgs(img_raw, crop_coords) # (numbox, 3, 32, 320)
        return list_crop_img

    def crop_imgs(self, img_raw, crop_coords):
        list_crop_img = []
        max_wh_ratio = 320 / 48
        for crop_idx, crop_coord in enumerate(crop_coords):
            crop_img = img_raw[crop_coord[0][1]:crop_coord[2][1], crop_coord[0][0]:crop_coord[1][0], ::-1].copy()
            list_crop_img.append(crop_img)
            h, w = crop_img.shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        norm_img_batch = []
        for crop_img in list_crop_img:
            norm_img = self.resize_norm_img(crop_img, max_wh_ratio)
            
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)

        norm_img_batch = np.concatenate(norm_img_batch)
        return norm_img_batch

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
