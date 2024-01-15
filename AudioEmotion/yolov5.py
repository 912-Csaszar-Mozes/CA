import torch
import cv2

class Yolov5SpectrogramEmotion:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights.pt', force_reload=False)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"device: {device}")
        self.model = self.model.to(device)
        self.model.eval()

    def evaluate(self, img_path):
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (480, 480))
        result = self.model(image)
        return result