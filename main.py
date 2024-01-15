from classes import * 
from data_providers import *
import time
import cv2


class Main:
    def __init__(self, data_provider, auto_start=True):
        self.face_recognition = FaceRecognition()
        self.super_resolution = SuperResolution()
        self.expression_recognition = ExpressionRecognition()
        self.transform = transforms.ToTensor()
        self.imaging = Imaging()
        self.provider = data_provider
        if auto_start:
            self.main()

    def get_result(self, img):
        img_tensor = self.transform(img)
        with torch.no_grad():
            faces, dets = self.face_recognition.get_output(img_tensor)
            faces = self.super_resolution.get_output(faces)
            results, predicted = self.expression_recognition.get_output(faces)
        return img_tensor, results, predicted, dets

    def show_result(self, img_tensor, results, predicted, dets):
        self.imaging.show_image_boxes(img_tensor, dets, f"Emotion of the crowd: {self.expression_recognition.class_names[int(ExpressionRecognition.prediction_from_score(results).numpy())]}")

    def main(self):
        while not self.provider.finished():
            self.show_result(*self.get_result(self.provider.next_img()))

dp = VideoProvider('test_data/Audience2.mp4')
Main(dp)


time.sleep(2)