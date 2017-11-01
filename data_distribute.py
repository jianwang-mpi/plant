
from PIL import Image
import os
import numpy as np
import random as rdm
image_array = []
new_width = 500
new_height = 500
for path in os.listdir('raw_data'):
    print('path: {0}'.format(path))
    for file in os.walk(os.path.join('raw_data', path)):
        print(file)
        image_files = file[2]
        rdm.shuffle(image_files)
        size = len(image_files)
        test_size = size // 4
        train_size = size - test_size
        i = 0
        for image_file in image_files:
            i += 1
            image_path = os.path.join('raw_data',path,image_file)
            image = Image.open(image_path)
            width, height = image.size  # Get dimensions

            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2

            image = image.crop((left, top, right, bottom))
            image.thumbnail((256,256))
            image = image.convert('RGB')
            # image_array.append(np.array(image))

            if i<=test_size:
                os.makedirs(os.path.join('dataset/test',path),exist_ok=True)
                image.save(os.path.join('dataset/test', path, image_file))
            else:
                os.makedirs(os.path.join('dataset/train', path), exist_ok=True)
                image.save(os.path.join('dataset/train', path, image_file))

'''
image_array = np.array(image_array)
mean = image_array.mean(axis=(0,1,2))
std = image_array.std(axis=(0,1,2))
print(mean/256)
print(std/256)
'''
