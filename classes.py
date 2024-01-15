import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage.transform import resize
import numpy as np
from PIL import Image
from AudioEmotion.yolov5 import Yolov5SpectrogramEmotion

from FaceRecognition.model.model import DetectionModel
from FaceRecognition import trainer
from ExpressionRecognition.models import VGG

from utilities import *

import sys
import os
from math import floor, ceil
import json


class FaceRecognition:
    @staticmethod
    def get_faces(tensor, dets):
        faces = []
        for det in dets:
            x_start, x_end = floor(det[0]), ceil(det[2]) + 1
            y_start, y_end = floor(det[1]), ceil(det[3]) + 1
            face = tensor[:,y_start:y_end,x_start:x_end]
            if face.shape[1] > 12 and face.shape[2] > 12:
                faces.append(face)
        return faces

    @staticmethod
    def filter_dets(dets, treshold=1):
        new_dets = []
        for det in dets:
            if det[4] >= treshold:
                new_dets.append(det)
        return new_dets
        
    def __init__(self):
        self.model = DetectionModel(num_templates=25)
        checkpoint = torch.load(os.path.join(os.getcwd(), "FaceRecognition", "weights.pth"))
        self.model.load_state_dict(checkpoint["model"])
        self.rf = {
            'size': [859, 859],
            'stride': [8, 8],
            'offset': [-1, -1]
        }
    
        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        ])
        
        self.templates = np.round_(np.array(json.load(open(os.path.join(os.getcwd(), "FaceRecognition", "datasets", "templates.json")))), decimals=8) 

    def get_output(self, input):
        dets = trainer.get_detections(self.model, input.unsqueeze(0), self.templates, self.rf, self.val_transforms, 0.03,
                                      0.3, device='cuda')
        dets = FaceRecognition.filter_dets(dets)
        faces = FaceRecognition.get_faces(input, dets)
        return faces, dets

class SuperResolution:
    def __init__(self):
        sys.path.append(os.path.join('SuperResolution', 'model'))
        self.model = torch.load(os.path.join(os.getcwd(), "SuperResolution", "weights.pth"), map_location=lambda storage, loc: storage)

    def get_output(self, faces):
        #for face in faces:
            #print(face.shape)
        return [self.model(face.unsqueeze(0)).squeeze() if (face.shape[1] < 48 or face.shape[2] < 48) else face for face in faces]

class ExpressionRecognition:
    def __init__(self):
        self.model = VGG('VGG19')
        checkpoint = torch.load(os.path.join(os.getcwd(), "ExpressionRecognition", "weights.pth"))
        self.model.load_state_dict(checkpoint)
        self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


    @staticmethod
    def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    @staticmethod
    def convert_to_facial_compatible(tensor):
        cut_size = 44
    
        transform_test = transforms.Compose([
            transforms.TenCrop(cut_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])
    
        tensor = tensor.permute(1, 2, 0).numpy()
        tensor = ExpressionRecognition.rgb2gray(tensor)
        tensor = resize(tensor, (48,48), mode='symmetric')
        tensor = np.floor(tensor * 255).astype(np.uint8)
        tensor = tensor[:, :, np.newaxis]    
        tensor = np.concatenate((tensor, tensor, tensor), axis=2)
        img = Image.fromarray(tensor)
        return transform_test(img)

    @staticmethod
    def facial_compatible_to_inputs(inputs):
        ncrops, c, h, w = np.shape(inputs)
        
        inputs = inputs.view(-1, c, h, w)
        return inputs, ncrops

    @staticmethod
    # Process facial recognition output
    def outputs_to_result(outputs, ncrops):
        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
        
        score = F.softmax(outputs_avg, -1)
        return score

    @staticmethod
    def prediction_from_score(score):
        _, predicted = torch.max(score.data, 0)
        return predicted

    def get_output(self, faces):
        results = torch.zeros(7)
        for face in faces:
            inputs = ExpressionRecognition.convert_to_facial_compatible(face)
            inputs, ncrops = ExpressionRecognition.facial_compatible_to_inputs(inputs)
            outputs = self.model(inputs)
            score = ExpressionRecognition.outputs_to_result(outputs, ncrops)
            results += score   
        predicted = ExpressionRecognition.prediction_from_score(results)
        return results, predicted

class AudioEmotion:
    def __init__(self):
        self.yolov5 = Yolov5SpectrogramEmotion()

    def get_result(self):
        return self.yolov5.evaluate(os.path.join('AudioEmotion', 'spectrogram.jpg'))