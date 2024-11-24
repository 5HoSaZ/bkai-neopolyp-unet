from torch.utils.data import Dataset
import numpy as np
import cv2


def one_hot_encode(mask):
    image = cv2.imread(mask_path)
    image = cv2.resize(image, self.resize)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 20])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 20])
    upper_red2 = np.array([179, 255, 255])


class NeoPolypDataset(Dataset):
    """Custom"""

    TRAIN_IMAGE_DIR = "./datasets/train/train"
    TRAIN_TARGET_DIR = "./datasets/train_gt/train_gt"
    TEST_IMAGE_DIR = "./datasets/test/test"

    def __init__(self):
        super().__init__()
