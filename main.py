from classes import * 
from data_providers import *
import time
import pickle
import subprocess
from BodyLanguageRecognition.main import *

class Main:
    def __init__(self, video_rel_path, skip_secs=1.0, auto_start=True):
        self.face_recognition = FaceRecognition()
        self.super_resolution = SuperResolution()
        self.expression_recognition = ExpressionRecognition()
        self.audio_emotion = AudioEmotion()
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
    
    def get_audio_results(self, spectrogram_db):
        binary_data_array = pickle.dumps(spectrogram_db)
        with open(os.path.join('AudioEmotion', 'spectrogram_db.pkl'), 'wb') as temp_file:
            temp_file.write(binary_data_array)
        try:
            subprocess.run(['python', 
                os.path.join('AudioEmotion', 'spectrogram_image.py'),
                '--sample_rate', 
                str(self.audio_spectrogram_provider.sample_rate)],
                check=True
            )
            return self.audio_emotion.get_result()
        except subprocess.CalledProcessError as e:
            print(f"Script execution failed with return code {e.returncode}")
            print(f"Error output:\n{e.output}")  
            return None

    def get_body_lang_results(self, to_process):
        return process(to_process)

    def show_results(self, img_tensor, results, predicted, dets, audio_predicted_emotion, audio_confidence, body_lang_emotion, body_lang_count):
        self.imaging.show_image_boxes(
            img_tensor, dets, 
            "Expression predicted emotion: "
            + str(self.expression_recognition.class_names[int(ExpressionRecognition.prediction_from_score(results).numpy())])
            + '\nBody language predicted emotion: ' + body_lang_emotion + " number of people: " + str(body_lang_count)
            + '\nAudio predicted emotion: ' + audio_predicted_emotion + " with confidence: " + str(audio_confidence)  
        )

    def main(self):
        sound_plays = False
        while not self.video_provider.finished() and not self.audio_spectrogram_provider.finished():
            predicted_label, confidence = self.get_audio_results(self.audio_spectrogram_provider.next_img())
            if predicted_label == -1:
                predicted_emotion = 'Unknown'
            else:
                predicted_emotion = self.audio_emotion.class_names[predicted_label]
            confidence = int(confidence * 100) / 100.0
            img = self.video_provider.next_img()
            body_lang_results = self.get_body_lang_results(img)
            body_lang_results = sorted(body_lang_results, key=lambda x: x[1], reverse=True)
            self.show_results(*self.get_facial_results(img), predicted_emotion, confidence, body_lang_results[0][0], body_lang_results[0][1])
            
            if sound_plays == False:
                self.audio_spectrogram_provider.play_audio_in_background()
                sound_plays = True

Main('test_data/bosnia_trimmed.mp4')


time.sleep(2)