from abc import ABC, abstractmethod
import os
from PIL import Image
import cv2
from moviepy.editor import *
from scipy.io import wavfile
import pyloudnorm
import librosa
import numpy as np
import matplotlib.pyplot as plt
import playsound

class DataProvider(ABC):
    @abstractmethod
    def next_img(self):
        # Returns a PIL image if there is one to return, None if the stream is empty
        pass

    @abstractmethod
    def finished(self):
        # return True if there is no more data to be sent, false otherwise
        pass

class ImageProvider(DataProvider):
    def __init__(self, img_rel_path):
        # based on a relative path image open the image
        self.img_path = img_rel_path
        self.done = False

    def next_img(self):
        if not self.finished():
            self.done = True
            return Image.open(os.path.join(os.getcwd(), self.img_path))

    def finished(self):
        return self.done

class ImagesProvider(DataProvider):
    def __init__(self, *img_rel_paths):
        # based on a relative path image open the image
        self.img_paths = img_rel_paths
        self.idx = 0

    def next_img(self):
        if not self.finished():
            img = Image.open(os.path.join(os.getcwd(), self.img_paths[self.idx]))
            self.idx += 1
            return img

    def finished(self):
        return self.idx >= len(self.img_paths)

class VideoProvider(DataProvider):
    def __init__(self, video_rel_path, skip_secs=0.2):
        self.video = cv2.VideoCapture(os.path.join(os.getcwd(), video_rel_path))
        fps = self.video.get(cv2.CAP_PROP_FPS)
        self.skip_frames = int(fps * skip_secs)
        self.current_img = None
        self.done = False

    def next_step(self):
        ret, frame = self.video.read()
        if not ret:
            self.done=True
            self.video.release()
        else:
            # get the current image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_img = Image.fromarray(rgb_frame)

            # move time forward
            current_frame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
            self.video.set(cv2.CAP_PROP_POS_FRAMES, current_frame + self.skip_frames)
            
    def next_img(self):
        self.next_step()
        if not self.finished():
            return self.current_img

    def finished(self):
        return self.done

class AudioSpectrogramProvider(DataProvider):
    def __init__(self, video_rel_path, skip_secs=1.0):
        videoclip = VideoFileClip(video_rel_path)
        self.audio_path = video_rel_path[:-4] + '.wav'
        audioclip = videoclip.audio
        audioclip.write_audiofile(self.audio_path)
        audioclip.close()
        videoclip.close()
        sample_rate, sample = wavfile.read(self.audio_path)
        if len(sample.shape) == 2: # stereo to mono
            sample = (sample[:, 0] + sample[:, 1]) / 2
        self.sample_rate = sample_rate
        self.sample = self.normalize_audio(sample, sample_rate)
        self.skip_frames = int(sample_rate * skip_secs)
        self.current_frame = 0
        self.current_img = None
        self.done = False

    def next_step(self):
        current_sample_end = self.current_frame + self.skip_frames
        if current_sample_end > len(self.sample):
            self.done = True
        else:
            current_sample = self.sample[self.current_frame:current_sample_end]
            self.current_img = self.sound_to_spectrogram(self.sample_rate, current_sample)
            self.current_frame += self.skip_frames

    def next_img(self):
            self.next_step()
            if not self.finished():
                return self.current_img

    def finished(self):
        return self.done

    def audio_loudness(self, audio, samplerate):
        meter = pyloudnorm.Meter(samplerate)
        loudness = meter.integrated_loudness(audio)
        return loudness

    def normalize_audio(self, audio, samplerate, new_loudness=-23.0):
        loudness = self.audio_loudness(audio, samplerate)
        normalized_audio = pyloudnorm.normalize.loudness(audio, loudness, new_loudness)
        return normalized_audio

    def sound_to_spectrogram(self, sample_rate, sample):
        spectrogram = librosa.feature.melspectrogram(y=sample.astype(float), 
                                                    sr=sample_rate, 
                                                    win_length=400,
                                                    hop_length=len(sample)//256,
                                                    n_mels=259,
                                                    n_fft=6000,
                                                    fmin=20, 
                                                    fmax=2e4)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        return spectrogram_db

    def play_audio_in_background(self):
        playsound.playsound(self.audio_path, False)
        