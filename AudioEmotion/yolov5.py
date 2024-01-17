import torch
import cv2
import os
import matplotlib
import pathlib

class Yolov5SpectrogramEmotion:
    def __init__(self):
        temp = pathlib.PosixPath # Comment this if you are on Linux
        pathlib.PosixPath = pathlib.WindowsPath #
        current_backend = matplotlib.get_backend() # yolov5 hides matplotlib so we need to restore it...
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join('AudioEmotion', 'weights.pt'), force_reload=True)
        matplotlib.use(current_backend)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"device: {device}")
        self.model = self.model.to(device)
        self.model.eval()

    def evaluate(self, img_path):
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (480, 480))
        result = self.model(image)
        return result