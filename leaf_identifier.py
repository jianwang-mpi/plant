from __future__ import print_function, division

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import models
from optparse import OptionParser
from data_loader import use_gpu, dataloders, dataset_sizes


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, save_file='result.txt'):
    global writer
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_step = 0
        test_step = 0

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            steps = 0
            running_corrects = 0

            iter_times = 0
            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    iter_times += 1
                    loss.backward()
                    optimizer.step()
                    train_step += inputs.data.size()[0]
                else:
                    test_step += inputs.data.size()[0]

                # statistics
                running_loss += loss.data[0]
                steps += 1
                running_corrects += torch.sum(preds == labels.data)
                # logging
                if phase == 'train' and train_step % 50 == 0:
                    writer.add_scalar(tag='train_loss', scalar_value=running_loss / steps,
                                      global_step=dataset_sizes[phase] * epoch + train_step)
                    running_loss = 0
                    steps = 0
                elif test_step % 50 == 0:
                    writer.add_scalar(tag='test_loss', scalar_value=loss.data[0],
                                      global_step=dataset_sizes[phase] * epoch + test_step)
                    running_loss = 0
                    steps = 0

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Acc: {:.4f}'.format(
                phase, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # write result to file
    with open(save_file, mode='w') as result_file:
        result_file.write('Training complete in {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        result_file.write('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


parser = OptionParser()
parser.add_option(
    '-m', '--model',
    action='store',
    dest='model_type',
    type='string',
    default='resnet'

)
options, args = parser.parse_args()
model_type = options.model_type
writer = SummaryWriter(log_dir=model_type)
if model_type == 'resnet':
    model_ft = models.resnet50(pretrained=True)
    fc_in_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(in_features=fc_in_features, out_features=184)
elif model_type == 'densenet':
    model_ft = models.densenet121(pretrained=True)
    fc_in_features = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(in_features=fc_in_features, out_features=184)
else:
    model_ft = models.inception_v3(pretrained=True)
    fc_in_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(in_features=fc_in_features, out_features=184)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25, save_file=os.path.join("result_folder", model_type))
