import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa

parser = argparse.ArgumentParser()

parser.add_argument('--sample_rate', type=int)
args = parser.parse_args()
sample_rate = args.sample_rate

spectrogram_db = []
with open('spectrogram_db.json', 'r') as file:
    json_array = file.read()
    spectrogram_db = np.array(json.loads(json_array))

plt.axis('off')
plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
librosa.display.specshow(spectrogram_db, x_axis='time',
                        y_axis='mel', sr=sample_rate,
                            cmap='jet', vmin=-116, vmax=10)
plt.savefig('spectrogram.jpg', pad_inches=0)