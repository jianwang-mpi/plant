
from PIL import Image
import os
import numpy as np
image_array = []
for path in os.listdir('raw_data'):
    print('path: {0}'.format(path))
    for file in os.walk(os.path.join('raw_data', path)):
        print(file)
        image_files = file[2]
        for image_file in image_files:
            image_path = os.path.join('raw_data',path,image_file)
            image = Image.open(image_path)
            image = image.resize()
            image_array.append(np.array(image))

image_array = np.array(image_array)
mean = image_array.mean()
std = image_array.std()
print(mean)
print(std)
