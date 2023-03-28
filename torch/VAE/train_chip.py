import torch
import torch.optim as optim
import torch.nn as nn
import model
import torchvision.transforms as transforms
import torchvision
import matplotlib
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from engine import train, validate
from utils import save_reconstructed_images, image_to_vid, save_loss_plot
from torch.utils.data import random_split
import numpy as np
matplotlib.use('Agg')
# matplotlib.style.use('ggplot')

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    # transforms.ToTensor(),
])

def get_data():
    # train_y = np.load('./data/y_train.npy').astype(np.float32)

    # train_y = np.load('./data/y_test.npy').astype(np.float32)

    train_y = np.load('./data/y_train_2000.npy').astype(np.float32)


    ##### Filter unrealistic cases #####
    idx_train = np.amax(train_y, axis=(1, 2)) < 300
    train_y = train_y[idx_train]
    ##### Filter unrealistic cases #####

    T_max = train_y.max()
    T_min = train_y.min()
    train_y = (train_y - T_min) / (T_max - T_min)

    train_y = np.expand_dims(train_y, -1)

    print('train_y shape: ', train_y.shape)

    train_y = torch.from_numpy(train_y)

    print('train_y shape: ', train_y.shape)
    # train_y = torch.permute(train_y, (0,3,1,2))
    train_y = train_y.permute((0,3,1,2))
    # train_y = transform(train_y)

    print('train_y shape: ', train_y.shape)
    dataset = torch.utils.data.TensorDataset(train_y)
    return dataset



DEVICE=4
device = torch.device(f'cuda:{DEVICE}' if torch.cuda.is_available() else 'cpu')
# initialize the model
model = model.ConvVAE().to(device)
# set the learning parameters
lr = 0.001
epochs = 1000
batch_size = 64
# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.AdamW(model.parameters(), lr=lr)

# criterion = nn.BCELoss(reduction='sum')
# criterion = nn.MSELoss(reduction='sum')
criterion = nn.L1Loss(reduction='sum')

# a list to save all the reconstructed images in PyTorch grid format
grid_images = []

dataset = get_data()




train_size = int(0.8*len(dataset))
# val_size = int(0.1*len(dataset))
# test_size = int(0.2*len(dataset))
test_size = len(dataset)-train_size

# train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
trainset, testset = random_split(dataset, [train_size, test_size])

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
# valloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)


# transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
# ])
# training set and train data loader
# trainset = torchvision.datasets.MNIST(
#     root='../input', train=True, download=True, transform=transform
# )
# trainloader = DataLoader(
#     trainset, batch_size=batch_size, shuffle=True
# )
# # validation set and validation data loader
# testset = torchvision.datasets.MNIST(
#     root='../input', train=False, download=True, transform=transform
# )
# testloader = DataLoader(
#     testset, batch_size=batch_size, shuffle=False
# )



train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, trainloader, trainset, device, optimizer, criterion
    )
    valid_epoch_loss, recon_images = validate(
        model, testloader, testset, device, criterion
    )
    # testNcompare(model, testloader, testset, device, criterion)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    # save the reconstructed images from the validation loop
    # save_reconstructed_images(recon_images, epoch+1)
    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())
    grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")
    # save the reconstructions as a .gif file
    # image_to_vid(grid_images)



# save the loss plots to disk
save_loss_plot(train_loss, valid_loss)
print('TRAINING COMPLETE')