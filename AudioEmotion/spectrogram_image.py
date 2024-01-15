import argparse
import pickle
import matplotlib.pyplot as plt
import librosa
import os

parser = argparse.ArgumentParser()

parser.add_argument('--sample_rate', type=int)
args = parser.parse_args()
sample_rate = args.sample_rate

spectrogram_db = []
with open(os.path.join('AudioEmotion', 'spectrogram_db.pkl'), 'rb') as file:
    binary_data_array = file.read()
    spectrogram_db = pickle.loads(binary_data_array)

plt.axis('off')
plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
librosa.display.specshow(spectrogram_db, x_axis='time',
                        y_axis='mel', sr=sample_rate,
                            cmap='jet', vmin=-116, vmax=10)
plt.savefig(os.path.join('AudioEmotion', 'spectrogram.jpg'), pad_inches=0)