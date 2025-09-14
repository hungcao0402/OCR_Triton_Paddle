import copy
import numpy as np
import cv2
import math

class RecPreprocessor:
    def __init__(self, rec_image_shape=[3, 48, 320], input_shape=None, max_imgW=1280):
        self.rec_image_shape = rec_image_shape
        self.crop = Crop()
        self.normalize = OCRReisizeNormImg(rec_image_shape, input_shape, max_imgW)
        self.batch = ToBatch()
    
    def __call__(self, img_raw, dt_boxes):
        crop_list = self.crop.run(img_raw, dt_boxes)
        if len(crop_list):
            norm_crops = self.normalize(crop_list)
            batch = self.batch(norm_crops)[0]
        else:
            batch = np.zeros((1,*self.rec_image_shape))
        return batch


class OCRReisizeNormImg:
    """for ocr image resize and normalization"""

    def __init__(self, rec_image_shape=[3, 48, 320], input_shape=None, max_imgW=1280):
        super().__init__()
        self.rec_image_shape = rec_image_shape
        self.input_shape = input_shape
        self.max_imgW = max_imgW

    def resize_norm_img(self, img, max_wh_ratio):
        """resize and normalize the img"""
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        if imgW > self.max_imgW:
            resized_image = cv2.resize(img, (self.max_imgW, imgH))
            resized_w = self.max_imgW
            imgW = self.max_imgW
        else:
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
    
    def __call__(self, imgs):
        """apply"""
        if self.input_shape is None:
            return [self.resize(img) for img in imgs]
        else:
            return [self.staticResize(img) for img in imgs]

    def resize(self, img):
        imgC, imgH, imgW = self.rec_image_shape
        max_wh_ratio = imgW / imgH
        h, w = img.shape[:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
        img = self.resize_norm_img(img, max_wh_ratio)
        return img

    def staticResize(self, img):
        imgC, imgH, imgW = self.input_shape
        resized_image = cv2.resize(img, (int(imgW), int(imgH)))
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        return resized_image

class ToBatch:
    """A class for batching and padding images to a uniform width."""

    def __pad_imgs(self, imgs):
        """Pad images to the maximum width in the batch.

        Args:
            imgs (list of np.ndarrays): List of images to pad.

        Returns:
            list of np.ndarrays: List of padded images.
        """
        max_width = max(img.shape[2] for img in imgs)
        padded_imgs = []
        for img in imgs:
            _, height, width = img.shape
            pad_width = max_width - width
            padded_img = np.pad(
                img,
                ((0, 0), (0, 0), (0, pad_width)),
                mode="constant",
                constant_values=0,
            )
            padded_imgs.append(padded_img)
        return padded_imgs

    def __call__(self, imgs):
        """Call method to pad images and stack them into a batch.

        Args:
            imgs (list of np.ndarrays): List of images to process.

        Returns:
            list of np.ndarrays: List containing a stacked tensor of the padded images.
        """
        imgs = self.__pad_imgs(imgs)
        return [np.stack(imgs, axis=0).astype(dtype=np.float32, copy=False)]
    
class Crop:
    
    def run(self, img_raw, dt_boxes):
        # crop_coords = [crop.astype(int) for crop in dt_boxes]
        # list_crop_img = self.crop_imgs(img_raw, crop_coords) # (numbox, 3, 32, 320)
        
        # if self.det_box_type == "quad":
        dt_boxes = np.array(dt_boxes)
        output_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_minarea_rect_crop(img_raw, tmp_box)
            output_list.append(img_crop)
        # elif self.det_box_type == "poly":
        #     output_list = []
        #     dt_boxes = dt_polys
        #     for bno in range(len(dt_boxes)):
        #         tmp_box = copy.deepcopy(dt_boxes[bno])
        #         img_crop = self.get_poly_rect_crop(img.copy(), tmp_box)
        #         output_list.append(img_crop)
                
        return output_list

    def get_minarea_rect_crop(self, img: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Get the minimum area rectangle crop from the given image and points.

        Args:
            img (np.ndarray): The input image.
            points (np.ndarray): A list of points defining the shape to be cropped.

        Returns:
            np.ndarray: The cropped image with the minimum area rectangle.
        """
        bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_a, index_b, index_c, index_d = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_a = 0
            index_d = 1
        else:
            index_a = 1
            index_d = 0
        if points[3][1] > points[2][1]:
            index_b = 2
            index_c = 3
        else:
            index_b = 3
            index_c = 2

        box = [points[index_a], points[index_b], points[index_c], points[index_d]]
        crop_img = self.get_rotate_crop_image(img, np.array(box))
        return crop_img

    def get_rotate_crop_image(self, img: np.ndarray, points: list) -> np.ndarray:
        """
        Crop and rotate the input image based on the given four points to form a perspective-transformed image.

        Args:
            img (np.ndarray): The input image array.
            points (list): A list of four 2D points defining the crop region in the image.

        Returns:
            np.ndarray: The transformed image array.
        """
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
