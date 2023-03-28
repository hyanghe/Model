import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import operator 
from Adam import Adam
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
from functools import reduce
from timeit import default_timer
from os.path import exists
import matplotlib
from torchsummary import summary
# from unet_model import UNet

matplotlib.use('agg')
cur_path = os.getcwd()
os.makedirs('./Unet', mode=0o777, exist_ok=True)
work_dir = os.getcwd() + '/Unet/'
os.makedirs(work_dir + 'ckpt', mode=0o777, exist_ok=True)
ckpt_dir = work_dir + 'ckpt/'


# #### EXP 2 ######
# TRAIN_DATA_X = 'x_train_random_scale2_5.npy'
# TRAIN_DATA_Y = 'y_train_random_scale2_5.npy'

# TEST_DATA_X = 'x_train_fix_scale2_5.npy'
# TEST_DATA_Y = 'y_train_fix_scale2_5.npy'

# #### EXP 3 ######
# TRAIN_DATA_X = 'x_train_fix_scale2_5.npy'
# TRAIN_DATA_Y = 'y_train_fix_scale2_5.npy'

# TEST_DATA_X = 'x_train_random_scale2_5.npy'
# TEST_DATA_Y = 'y_train_random_scale2_5.npy'

#### EXP 4 ######
TRAIN_DATA_X = 'x_train.npy'
TRAIN_DATA_Y = 'y_train.npy'

TEST_DATA_X = 'x_test.npy'
TEST_DATA_Y = 'y_test.npy'


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation1 = nn.ReLU(inplace=False)

        self.convolution2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation2 = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.activation1(self.convolution1(x))
        x = self.activation2(self.convolution2(x))
        return x

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvolution(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return x, skip


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConvolution(out_channels+out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # print('x, skip: ', x.shape, skip.shape)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, filters):
        super().__init__()
        num_c = filters
        self.down1 = DownSample(1, num_c)
        self.down2 = DownSample(num_c, num_c*2)
        self.down3 = DownSample(num_c*2, num_c*4)
        self.down4 = DownSample(num_c*4, num_c*8)

        self.z = DoubleConvolution(num_c*8, num_c*16)

        self.up1 = UpSample(num_c*16, num_c*8)
        self.up2 = UpSample(num_c*8, num_c*4)
        self.up3 = UpSample(num_c*4, num_c*2)
        self.up4 = UpSample(num_c*2, num_c*1)

        self.outputs = nn.Conv2d(num_c, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # print('x shape is: ', x.shape)
        x1, skip1 = self.down1(x)
        # print('x1, skip1: ', x1.shape, skip1.shape)
        x2, skip2 = self.down2(x1)
        # print('x2, skip2: ', x2.shape, skip2.shape)
        x3, skip3 = self.down3(x2)
        # print('x3, skip3: ', x3.shape, skip3.shape)
        x4, skip4 = self.down4(x3)
        # print('x4, skip4: ', x4.shape, skip4.shape)

        z = self.z(x4)
        # print('z shape: ', z.shape)
        y1 = self.up1(z, skip4)
        y2 = self.up2(y1, skip3)
        y3 = self.up3(y2, skip2)
        y4 = self.up4(y3, skip1)

        return self.outputs(y4)


def mape(true, pred, T_min, T_max): 
    true, pred = np.array(true), np.array(pred)
    pred = pred * (T_max - T_min) + T_min
    true = true * (T_max - T_min) + T_min
    mask = true != 0
    return np.mean(np.abs((true - pred) / true)[mask])

def relativeL2(true, pred, T_min, T_max):
    pred = pred * (T_max - T_min) + T_min
    true = true * (T_max - T_min) + T_min
    # return np.linalg.norm(true.flatten() - pred.numpy().flatten()) / np.linalg.norm(true.flatten()) 
    return np.linalg.norm(true.flatten() - pred.flatten()) / np.linalg.norm(true.flatten()) 

def mae(true, pred, T_min, T_max):
    print('pred true', pred.shape, true.shape)
    pred = pred * (T_max - T_min) + T_min
    true = true * (T_max - T_min) + T_min
    return np.mean(np.abs(true - pred))


# Data
def get_data(batch_size):
    print('==> Preparing data..')
    val_percent = 0.1
    # batch_size = 100

    dir_data = './data/'
    # train_x = np.load(dir_data + 'x_train.npy').astype(np.float32)
    # train_y = np.load(dir_data + 'y_train.npy').astype(np.float32)
    train_x = np.load(dir_data + TRAIN_DATA_X).astype(np.float32)
    train_y = np.load(dir_data + TRAIN_DATA_Y).astype(np.float32)





    ##### Filter unrealistic cases #####
    idx_train = np.amax(train_y, axis=(1, 2)) < 300
    train_x = train_x[idx_train]
    train_y = train_y[idx_train]
    ##### Filter unrealistic cases #####

    Power_max = train_x.max()
    Power_min = train_x.min()
    T_max = train_y.max()
    T_min = train_y.min()
    train_x = (train_x - Power_min) / (Power_max - Power_min)
    train_y = (train_y - T_min) / (T_max - T_min)

    train_x = np.expand_dims(train_x, -1)
    train_x = torch.from_numpy(train_x)
    train_y = np.expand_dims(train_y, -1)
    train_y = torch.from_numpy(train_y)

    print('train_x shape: ', train_x.shape)
    print('train_y shape: ', train_y.shape)
    # train_x = torch.permute(train_x, (0,3,1,2))
    # train_y = torch.permute(train_y, (0,3,1,2))

    train_x = train_x.permute((0,3,1,2))
    train_y = train_y.permute((0,3,1,2))


    # test_x = np.load(dir_data + 'x_test.npy').astype(np.float32)
    # test_y = np.load(dir_data + 'y_test.npy').astype(np.float32)
    test_x = np.load(dir_data + TEST_DATA_X).astype(np.float32)
    test_y = np.load(dir_data + TEST_DATA_Y).astype(np.float32)
 
    idx_test = np.amax(test_y, axis=(1, 2)) < 300
    test_x = test_x[idx_test]
    test_y = test_y[idx_test]

    test_x = (test_x - Power_min) / (Power_max - Power_min)
    test_y = (test_y - T_min) / (T_max - T_min)

    test_x = np.expand_dims(test_x, -1)
    test_x = torch.from_numpy(test_x)
    test_y = np.expand_dims(test_y, -1)
    test_y = torch.from_numpy(test_y)

    print('test_x shape: ', test_x.shape)
    print('test_y shape: ', test_y.shape)
    # test_x = torch.permute(test_x, (0,3,1,2))
    # test_y = torch.permute(test_y, (0,3,1,2))

    test_x = test_x.permute((0,3,1,2))
    test_y = test_y.permute((0,3,1,2))

    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)

    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    # print('train_set: ', train_set)
    # print('val_set: ', val_set)
    # raise
    # 3. Create data loaders

    # val_set = test_dataset


    # loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    loader_args = dict(batch_size=batch_size)
    trainloader = DataLoader(train_set, shuffle=True, **loader_args)
    valloader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    testloader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)
    return trainloader, valloader, testloader, T_min, T_max, test_x, test_y


# def test_net(model,
#               device,
#               train_loader,
#               test_loader,
#               x_normalizer,
#               y_normalizer,
#               batch_size: int = 20,
#               s: int=80
#               ):
def test_net(model,
              DEVICE_NUM,
              testloader,
              T_min,
              T_max,
              input_size,
              batch_size
              ):
    with torch.no_grad():
        true = []
        pred = []

        for x, y in testloader:
            x, y = x.cuda(DEVICE_NUM), y.cuda(DEVICE_NUM)

            out = model(x).reshape(batch_size, input_size, input_size)

            out = out * (T_max - T_min) + T_min
            y = y * (T_max - T_min) + T_min


            # print('y shape: ', y.detach().cpu().numpy().shape)
            # raise
            true.extend(y.detach().cpu().numpy())
            pred.extend(out.detach().cpu().numpy())
        true = np.squeeze(np.asarray(true))
        pred = np.squeeze(np.asarray(pred))

        mae = np.mean(np.abs(true - pred))
        # print('true.shape[0]', true.shape[0])
        # raise
        # idx_ls = np.random.choice(true.shape[0], 10)
        # print('idx_ls: ', idx_ls)
        # raise
        # print('true.shape', true.shape)
        # print('pred.shape', pred.shape)
        # raise
        idx_ls = np.arange(10)
        for idx in idx_ls:
            fig = plt.figure(figsize=(15, 5))
            plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.85, wspace=0.2, hspace=0.3)
            ax = fig.add_subplot(131)
            ax.set_title(f'Truth')
            im = ax.imshow(true[idx, :,:], origin='lower', cmap='jet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="7%", pad="2%")
            cb = fig.colorbar(im, cax=cax)

            ax = fig.add_subplot(132)
            im = ax.imshow(pred[idx, :,:], cmap='jet', origin='lower')
            ax.set_title(f'Pred')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="7%", pad="2%")
            cb = fig.colorbar(im, cax=cax)

            ax = fig.add_subplot(133)
            im = ax.imshow(abs(pred[idx, :,:] - true[idx, :,:]), cmap='jet', origin='lower')
            ax.set_title(f'Error')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="7%", pad="2%")
            cb = fig.colorbar(im, cax=cax)
            # plt.savefig(f'./figs/{cnt}.jpg')
            plt.savefig(f'./figs/final_test_sample_{idx}.jpg')
            # plt.show()
            plt.close()
            # cnt += 1
        
        # torch.save(net.state_dict(), "./checkpoint/network.pt")

        rel_l2 = np.linalg.norm(true.flatten() - pred.flatten()) / np.linalg.norm(true.flatten()) 
        mape_error = mape(true, pred, T_min, T_max)
        print('mae: ', mae)
        print('rel_l2: ', rel_l2)
        print('mape_error: ', mape_error)
        with open('./final_test_l2.txt', 'w') as f:
            f.write(f'mae is: {mae} \n')
            f.write(f'relative l2 is: {rel_l2} \n')
            f.write(f'mape_error is: {mape_error} \n')
################################################################
# training and evaluation
################################################################

def get_args():
    parser = argparse.ArgumentParser(description='FNO')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=256, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=3e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--mode', '-m', type=str, default='test', help='train or test')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU index')
    return parser.parse_args()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':
    args = get_args()

    DEVICE_NUM = args.gpu
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    mode = args.mode



    # DEVICE_NUM = 0
    # batch_size = 100
    # learning_rate = 3e-4
    # epochs = 5000
    # # mode = 'train'
    # mode = 'test'
    
    device = torch.device(f'cuda:{DEVICE_NUM}')

    ntrain = 5000
    ntest = 1000
    # s = 80
    # input_size = 80
    input_size = 160

    trainloader, valloader, testloader, T_min, T_max, test_x, test_y  = get_data(batch_size)
    
    # step_size = 100
    step_size = 1000000
    gamma = 0.5

    
    num_filter = 64


    model = Unet(num_filter).cuda(DEVICE_NUM)

    # model = UNet(n_channels=1, n_classes=1, bilinear=False).cuda(DEVICE_NUM)
# 
    # model = torch.nn.DataParallel(model, device_ids = [0,2,5, 7])
    model = torch.nn.DataParallel(model, device_ids = [1,3])

    print(count_params(model))
    summary(model, (1, 160, 160))
    # raise

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)

    # optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # myloss = LpLoss(size_average=False)
    # y_normalizer.cuda(DEVICE_NUM)
    criterion = nn.MSELoss()
    min_val_l2 = 10**10


    model_path = "./checkpoint/network.pt"
    file_exists = exists(model_path)
    if file_exists:
        print('model exist')
        model.load_state_dict(torch.load(model_path, map_location=device))
        num_params = count_params(model)
        print(f'model loaded, num_params: {num_params}')
    else:
        print('model does not exist!')

    if mode == 'train':
        for ep in range(epochs):
            model.train()
            t1 = default_timer()
            train_l2 = 0
            for x, y in trainloader:
                x, y = x.cuda(DEVICE_NUM), y.cuda(DEVICE_NUM)

                optimizer.zero_grad()
                out = model(x)
                # out = model(x).reshape(batch_size, out_channels, input_size, input_size)

                # out = (out - T_min) / (T_max - T_min)
                # y = (y - T_min) / (T_max - T_min)

                loss = criterion(out.view(batch_size,-1), y.view(batch_size,-1))
                loss.backward()

                optimizer.step()
                train_l2 += loss.item()

            scheduler.step()

            model.eval()

            val_l2 = 0.0
            with torch.no_grad():
                for x, y in valloader:
                    x, y = x.cuda(DEVICE_NUM), y.cuda(DEVICE_NUM)

                    out = model(x).reshape(batch_size, input_size, input_size)
                    # out = (out - T_min) / (T_max - T_min)

                    val_l2 += criterion(out.view(batch_size,-1), y.view(batch_size,-1)).item()

            train_l2/= len(trainloader)*trainloader.batch_size
            val_l2 /= len(valloader)*valloader.batch_size

            t2 = default_timer()
            cur_lr = get_lr(optimizer)
            print(f'epoch: {ep}, time: {t2-t1:.5f}, train_l2: {train_l2}, val_l2: {val_l2}, lr: {cur_lr}')

            with torch.no_grad():
                if val_l2 < min_val_l2:
                    idx = np.random.choice(len(test_x))
                    x_p, y_p = test_x[idx:idx+1].cuda(DEVICE_NUM), test_y[idx:idx+1].cuda(DEVICE_NUM)
                    pred_p = model(x_p).reshape(1, input_size, input_size)

                    pred_p = pred_p * (T_max - T_min) + T_min
                    y_p = y_p * (T_max - T_min) + T_min

                    pred_p = np.squeeze(pred_p.detach().cpu().numpy())
                    true_p = np.squeeze(y_p.detach().cpu().numpy())
                    # print('pred shape: ', pred.shape)
                    # print('true shape: ', true.shape)
                    # raise
                    fig = plt.figure(figsize=(15, 5))
                    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.85, wspace=0.2, hspace=0.3)
                    ax = fig.add_subplot(131)
                    ax.set_title(f'Truth')
                    im = ax.imshow(true_p, origin='lower', cmap='jet')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad="2%")
                    cb = fig.colorbar(im, cax=cax)

                    ax = fig.add_subplot(132)
                    im = ax.imshow(pred_p, cmap='jet', origin='lower')
                    ax.set_title(f'Pred')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad="2%")
                    cb = fig.colorbar(im, cax=cax)

                    ax = fig.add_subplot(133)
                    im = ax.imshow(abs(pred_p - true_p), cmap='jet', origin='lower')
                    ax.set_title(f'Error')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad="2%")
                    cb = fig.colorbar(im, cax=cax)
                    plt.savefig(f'./figs/case_{idx}.jpg')
                    # plt.show()
                    plt.close()
                    min_val_l2 = val_l2
                    torch.save(model.state_dict(), "./checkpoint/network.pt")
                    # print('pred shape: ', pred.shape)
                    # print('true shape: ', true.shape)
                    # raise
                    # network.load_state_dict(torch.load("./checkpoint/model.pt"))
    elif mode == 'test':
        model.eval()
        test_net(model,
              DEVICE_NUM,
              testloader,
              T_min,
              T_max,
              input_size,
              batch_size
              )