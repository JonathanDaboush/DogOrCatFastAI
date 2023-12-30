import os

from fastai.vision.all import *
from fastai.vision import *

import pandas as pd
if __name__ == '__main__':
    DATASET_PATH = 'Data'
    # to get class names.
    class_names = []
    for folder_name in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder_name)
        if os.path.isdir(folder_path):  # Check if it's a directory
            class_names.append(folder_name)

    image_paths = []
    labels = []

    # Supported image extensions
    valid_image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

    for i in range(len(class_names)):
        class_name = class_names[i]
        class_path = os.path.join(DATASET_PATH, class_name)

        for image_name in os.listdir(class_path):
            # Check if the file is an image
            _, extension = os.path.splitext(image_name)
            if extension.lower() in valid_image_extensions:
                image_path = os.path.join(class_path, image_name)
                image_paths.append(image_path)
                labels.append(i)

    df = pd.DataFrame(
        {
            'name': image_paths,
            'label': labels
        }
    )

    df.to_csv(
        path_or_buf=f'{DATASET_PATH}/labels.csv',
        index=False
    )

    dls = ImageDataLoaders.from_folder(
        path=DATASET_PATH,
        item_tfms=Resize(224),
        bs=16,
        batch_tfms=[Normalize.from_stats(*imagenet_stats), RandTransform()],
        valid_pct=0.2,
        num_workers=0  # Add this line
    )

    model = vision_learner(
        dls=dls,
        arch=models.resnet50,
        metrics=[accuracy, error_rate]
    )

    model.fine_tune(
        epochs=3
    )

    model.export('ImageDifferentiator.pkl')
    new_model = load_learner(f'{DATASET_PATH}/ImageDifferentiator.pkl')
    
    print(new_model.predict(
    item='download (4).jpg'
    ))
