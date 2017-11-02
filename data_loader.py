from torchvision import transforms, datasets
import torch
import os
data_transforms = {
    'train': transforms.Compose(
        [
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.43030695, 0.47506287, 0.4381275], std=[0.24648066, 0.23345278, 0.2778161])
        ]
    ),
    'test': transforms.Compose(
        [
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.43030695, 0.47506287, 0.4381275], std=[0.24648066, 0.23345278, 0.2778161])
        ]
    )

}
data_dir = 'dataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes


print(class_names[130])
print(class_names[134])
print(class_names[150])
print(class_names[122])
print(class_names[86])
use_gpu = torch.cuda.is_available()