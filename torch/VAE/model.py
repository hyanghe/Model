import torch
import torch.nn as nn
import torch.nn.functional as F

# kernel_size = 4 # (4, 4) kernel
kernel_size = 3 # (4, 4) kernel

init_channels = 8 # initial number of filters
image_channels = 1 # MNIST images are grayscale
latent_dim = 16 # latent dimension for sampling



# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        # self.enc4 = nn.Conv2d(
        #     in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, 
        #     stride=2, padding=0
        # )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64*10*10, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64*10*10)
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        # self.dec1 = nn.ConvTranspose2d(
        #     in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size, 
        #     stride=1, padding=0
        # )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        # print('x shape: ', x.shape)
        # encoding
        x = F.relu(self.enc1(x))
        # print('After 1st conv, x shape: ', x.shape);raise
        x = F.relu(self.enc2(x))
        # print('After 2 conv, x shape: ', x.shape)
        x = F.relu(self.enc3(x))
        # print('After 3 conv, x shape: ', x.shape)
        x = F.relu(self.enc4(x))
        # print('After 4 conv, x shape: ', x.shape)
        batch, _, _, _ = x.shape
        # print(' F.adaptive_avg_pool2d(x, 1) shape: ',  F.adaptive_avg_pool2d(x, 1).shape);raise
        bottleneck_dim = x.shape[-1]
        x = F.adaptive_avg_pool2d(x, bottleneck_dim).reshape(batch, -1) # from pdb import set_trace; set_trace()
        # print('After adaptive_avg_pool2d, x shape: ', x.shape);raise
        hidden = self.fc1(x)
        # print('hidden shape: ', hidden.shape)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        # print('mu shape: ', mu.shape)
        log_var = self.fc_log_var(hidden)
        # print('log_var shape: ', log_var.shape)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        # print('After reparameterize, z shape: ', z.shape)
        z = self.fc2(z)
        # print('After fc2, z shape: ', z.shape)
        # z = z.view(-1, 64, 1, 1)
        z = z.view(-1, 64, bottleneck_dim, bottleneck_dim)
        # print('z shape: ', z.shape)
 
        # decoding
        x = F.relu(self.dec1(z, output_size=(bottleneck_dim*2, bottleneck_dim*2)))
        # print('After 1st conv: ', x.shape)
        x = F.relu(self.dec2(x, output_size=(bottleneck_dim*4, bottleneck_dim*4)))
        # print('After 2 conv: ', x.shape)
        x = F.relu(self.dec3(x, output_size=(bottleneck_dim*8, bottleneck_dim*8)))
        # print('After 3 conv: ', x.shape)
        reconstruction = torch.sigmoid(self.dec4(x, output_size=(bottleneck_dim*16, bottleneck_dim*16)))
        # print('final reconstruction: ', reconstruction.shape)
        # print('Return reconstruction, mu, log_var shape: ', reconstruction.shape, mu.shape, log_var.shape);raise
        return reconstruction, mu, log_var