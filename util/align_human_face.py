import numpy as np
import PIL
import PIL.Image
import scipy
import scipy.ndimage
import dlib
import copy
from PIL import Image

from util.utils import tensor_to_numpy, numpy_to_tensor


def detect_face_dlib(self, img):
    # tensor to numpy array rgb uint8
    img = tensor_to_numpy(img)
    aligned_image, crop, rect = align_human_face(img=img,
                                                 detector=self.dlib_cnn_face_detector,
                                                 predictor=self.dlib_predictor,
                                                 output_size=512)

    aligned_image = np.array(aligned_image)
    aligned_image = numpy_to_tensor(aligned_image)
    return aligned_image, crop, rect


def get_landmark(img, detector, predictor):
    """get landmark with dlib
    :param detector: pretrained human face detector
    :param predictor: pretrained human face landmarks predictor
    :return: np.array shape=(68, 2)
    """
    # detector = dlib.get_frontal_face_detector()
    # dets, _, _ = detector.run(img, 1, -1)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d.rect)
    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)

    # face rect
    face_rect = [dets[0].rect.left(), dets[0].rect.top(), dets[0].rect.right(), dets[0].rect.bottom()]
    return lm, face_rect


def align_human_face(img, detector, predictor, type="tar", output_size=256):
    """
    :param img: numpy array of image
    :param detector: pretrained human face detector
    :param predictor: pretrained human face landmarks predictor
    :return: PIL Image
    """
    img_cp = copy.deepcopy(img)

    try:
        lm, face_rect = get_landmark(img, detector, predictor)
    except:
        return None, None

    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    # opencv to PIL
    img = PIL.Image.fromarray(img_cp)
    img.save(f"./debug/original-{type}.jpg")

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    # crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
    #         min(crop[3] + border, img.size[1]))
    # img.save("debug/raw.jpg")
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        # img = img.crop(crop)
        # masking except for face
        mask = np.zeros_like(img_cp)
        img_size = img.size[0]
        mask[max(crop[1], 0):min(crop[3], img_size), max(crop[0], 0):min(crop[2], img_size)] = 1.
        img = mask * img
        img = PIL.Image.fromarray(img)
        quad -= crop[0:2]
        img.save(f"./debug/resize-{type}.jpg")
        img = np.asarray(img)
        return img, mask
    else:
        img.save(f"./debug/resize-{type}.jpg")
        img = np.asarray(img)
        return img, None

# For debug
if __name__ == "__main__":
    dlib_predictor = dlib.shape_predictor('./pretrained_models/shape_predictor_68_face_landmarks.dat')
    dlib_cnn_face_detector = dlib.cnn_face_detection_model_v1(
        "../pretrained_models/mmod_human_face_detector.dat")
    align_human_face("./yerinbaek/full_body/00087.jpg", predictor=dlib_predictor, detector=dlib_cnn_face_detector, output_size=512)