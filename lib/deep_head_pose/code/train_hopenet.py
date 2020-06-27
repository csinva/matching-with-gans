import sys, os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import datasets, hopenet, hopelessnet
import torch.utils.model_zoo as model_zoo

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs', 
        help='Maximum number of training epochs.',
        default=50, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=64, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.000001, type=float)
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.', 
        default='Pose_300W_LP', type=str)
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='datasets/300W_LP', type=str)
    parser.add_argument(
        '--filename_list', dest='filename_list', 
        help='Path to text file containing relative paths for every example.',
        default='datasets/300W_LP/files.txt', type=str)
    parser.add_argument(
        '--output_string', dest='output_string', 
        help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument(
        '--alpha', dest='alpha', help='Regression loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--arch', dest='arch', 
        help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], '
            'ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

def get_ignored_params(model, arch):
    # Generator function that yields ignored params.
    if arch.find('ResNet') >= 0:
        b = [model.conv1, model.bn1, model.fc_finetune]
    elif arch.find('Squeezenet') >= 0 or arch.find('MobileNetV2') >= 0:
        b = [model.features[0]]
    else:
        raise('Invalid architecture is passed!')

    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_non_ignored_params(model, arch):
    # Generator function that yields params that will be optimized.
    if arch.find('ResNet') >= 0:
        b = [model.layer1, model.layer2, model.layer3, model.layer4]
    elif arch.find('Squeezenet') >= 0 or arch.find('MobileNetV2') >= 0:
        b = [model.features[1:]]
    else:
        raise('Invalid architecture is passed!')

    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_fc_params(model, arch):
    # Generator function that yields fc layer params.
    if arch.find('ResNet') >= 0:
        b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    elif arch.find('Squeezenet') >= 0 or arch.find('MobileNetV2') >= 0:
        b = [
            model.classifier_yaw, 
            model.classifier_pitch, 
            model.classifier_roll
        ]
    else:
        raise('Invalid architecture is passed!')

    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    # Network architecture
    if args.arch == 'ResNet18':
        model = hopenet.Hopenet(
            torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 66)
        pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    elif args.arch == 'ResNet34':
        model = hopenet.Hopenet(
            torchvision.models.resnet.BasicBlock, [3,4,6,3], 66)
        pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    elif args.arch == 'ResNet101':
        model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], 66)
        pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    elif args.arch == 'ResNet152':
        model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], 66)
        pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    elif args.arch == 'Squeezenet_1_0':
        model = hopelessnet.Hopeless_Squeezenet(args.arch, 66)
        pre_url = \
            'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth'
    elif args.arch == 'Squeezenet_1_1':
        model = hopelessnet.Hopeless_Squeezenet(args.arch, 66)
        pre_url = \
            'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth'
    elif args.arch == 'MobileNetV2':
        model = hopelessnet.Hopeless_MobileNetV2(66, 1.0)
        pre_url = \
            'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
    else:
        if args.arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

    if args.snapshot == '':
        load_filtered_state_dict(model, model_zoo.load_url(pre_url))
    else:
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize(240),
        transforms.RandomCrop(224), transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
            )
        ])

    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets.Pose_300W_LP_random_ds(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Synhead':
        pose_dataset = datasets.Synhead(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(
            args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(
            args.data_dir, args.filename_list, transformations)
    else:
        print('Error: not a valid dataset name')
        sys.exit()

    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)

    model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    # Regression loss coefficient
    alpha = args.alpha

    softmax = nn.Softmax(dim=1).cuda(gpu)
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

    optimizer = torch.optim.Adam([
        {'params': get_ignored_params(model, args.arch), 'lr': 0},
        {'params': get_non_ignored_params(model, args.arch), 'lr': args.lr},
        {'params': get_fc_params(model, args.arch), 'lr': args.lr * 5}
        ], lr = args.lr)

    print('Ready to train network.')
    for epoch in range(num_epochs):
        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = Variable(images).cuda(gpu)

            # Binned labels
            label_yaw = Variable(labels[:,0]).cuda(gpu)
            label_pitch = Variable(labels[:,1]).cuda(gpu)
            label_roll = Variable(labels[:,2]).cuda(gpu)

            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:,0]).cuda(gpu)
            label_pitch_cont = Variable(cont_labels[:,1]).cuda(gpu)
            label_roll_cont = Variable(cont_labels[:,2]).cuda(gpu)

            # Forward pass
            yaw, pitch, roll = model(images)

            # Cross entropy loss
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            # MSE loss
            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)

            yaw_predicted = \
                torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = \
                torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = \
                torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            grad_seq = \
                [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: '
                    'Yaw %.4f, Pitch %.4f, Roll %.4f'%(
                        epoch+1, 
                        num_epochs, 
                        i+1, 
                        len(pose_dataset)//batch_size, 
                        loss_yaw.item(), 
                        loss_pitch.item(), 
                        loss_roll.item()
                    )
                )

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...',
                torch.save(model.state_dict(),
                'output/snapshots/' + args.output_string + 
                '_epoch_'+ str(epoch+1) + '.pkl')
            )
