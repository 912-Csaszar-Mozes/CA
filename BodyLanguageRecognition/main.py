from ultralytics import YOLO
import pandas as pd
import os
from catboost import CatBoostClassifier
import torch

body_parts = {
    0: 'Nose',
    1: 'Left-eye',
    2: 'Right-eye',
    3: 'Left-ear',
    4: 'Right-ear',
    5: 'Left-shoulder',
    6: 'Right-shoulder',
    7: 'Left-elbow',
    8: 'Right-elbow',
    9: 'Left-wrist',
    10: 'Right-wrist',
    11: 'Left-hip',
    12: 'Right-hip',
    13: 'Left-knee',
    14: 'Right-knee',
    15: 'Left-ankle',
    16: 'Right-ankle'
}

common_body_parts = [
    'Right-hip', 'Right-knee', 'Right-ankle',
    'Left-hip', 'Left-knee', 'Left-ankle',
    'Nose',
    'Right-shoulder', 'Right-elbow', 'Right-wrist',
    'Left-shoulder', 'Left-elbow', 'Left-wrist',
]

common_body_parts = [f'{part}_x' for part in common_body_parts] + [f'{part}_y' for part in common_body_parts]

def process(image):
    script_directory = os.path.dirname(os.path.realpath(__file__))
    models_folder_path = os.path.join(script_directory, 'models')
    loaded_catboost_model = CatBoostClassifier()
    loaded_catboost_model.load_model(os.path.join(models_folder_path, 'body_emotion'))

    torch.cuda.set_device(0)
    model = YOLO('yolov8m-pose.pt')
    model.to('cuda')

    results = model(source = image, show=False, conf=0.3, save=False)

    df = pd.DataFrame(columns=common_body_parts)

    rows_to_append = []

    for result in results:
        row_data = {}
        for idx, keypoint in enumerate(result.keypoints.xyn[0]):
            body_part = body_parts.get(idx, f'Unknown-{idx}')
            row_data[f'{body_part}_x'] = keypoint[0].item()
            row_data[f'{body_part}_y'] = keypoint[1].item()

        rows_to_append.append(row_data)

    df = pd.DataFrame(rows_to_append)

    # df = df.replace(0, np.nan)
    # df = df.dropna(axis=0, how='all')

    kept_df = df[common_body_parts]
    predictions = loaded_catboost_model.predict(data=kept_df)

    unique_values = set(tuple(row) for row in predictions)

    print("Body language emotion:")

    for value in unique_values:
        count = predictions.tolist().count(list(value))
        print(f"Value: {value[0]}, Count: {count}")

    return [(value[0], predictions.tolist().count(list(value))) for value in unique_values]

# movies = os.path.join(script_directory, 'BoLD_dataset_sample\\')
#
# group = "0163.mp4"
# guy_walk = "0040.mp4"
# kids_mom = "0025.mp4"



