from abc import ABC, abstractmethod
import os
from PIL import Image
import cv2

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
        