from classes import * 
from data_providers import *
import time
import cv2
import json
import subprocess

class Main:
    def __init__(self, video_rel_path, skip_secs=1.0, auto_start=True):
        self.face_recognition = FaceRecognition()
        self.super_resolution = SuperResolution()
        self.expression_recognition = ExpressionRecognition()
        self.transform = transforms.ToTensor()
        self.imaging = Imaging()
        self.video_provider = VideoProvider(video_rel_path, skip_secs)
        self.audio_spectrogram_provider = AudioSpectrogramProvider(video_rel_path, skip_secs)
        if auto_start:
            self.main()

    def get_facial_results(self, img):
        img_tensor = self.transform(img)
        with torch.no_grad():
            faces, dets = self.face_recognition.get_output(img_tensor)
            faces = self.super_resolution.get_output(faces)
            results, predicted = self.expression_recognition.get_output(faces)
        return img_tensor, results, predicted, dets

    def show_results(self, img_tensor, results, predicted, dets):
        self.imaging.show_image_boxes(
            img_tensor, dets, 
            "Emotion of the crowd: " + 
            str(self.expression_recognition.class_names[int(ExpressionRecognition.prediction_from_score(results).numpy())]))

    def main(self):
        while not self.video_provider.finished() and not self.audio_spectrogram_provider.finished():
            spectrogram_db = self.audio_spectrogram_provider.next_img()
            json_array = json.dumps(spectrogram_db.tolist())
            with open('spectrogram_db.json', 'w') as temp_file:
                temp_file.write(json_array)
            try:
                subprocess.run(['python', 'spectrogram_image.py', '--sample_rate', str(self.audio_spectrogram_provider.sample_rate)],
                    check=True)
            except subprocess.CalledProcessError as e:
                print(f"Script execution failed with return code {e.returncode}")
                print(f"Error output:\n{e.output}")                               

            self.show_results(*self.get_facial_results(self.video_provider.next_img()))

Main('test_data/DenmarkVsEnglandTrimmed.mp4')


time.sleep(2)