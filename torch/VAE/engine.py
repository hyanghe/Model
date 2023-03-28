from tqdm import tqdm
import torch 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print('BCE, KLD: ', BCE, KLD);raise;
    return BCE + KLD

def train(model, dataloader, dataset, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data[0] # from pdb import set_trace; set_trace()
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter 
    return train_loss

def validate(model, dataloader, dataset, device, criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data= data[0]
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction


                orig_images = data[0,0,:,:].cpu()
                recon_images_sample = reconstruction[0,0,:,:].cpu()

                fig = plt.figure(figsize = (10, 3), dpi=100)
                plt.rcParams.update({'font.size': 15})
                plt.subplots_adjust(wspace=1.2, hspace = 0.5, left = 0.05, right=0.95, top=0.95, bottom=0.05)

                ax = fig.add_subplot(1, 3, 1)
                im = ax.imshow(orig_images, cmap='jet')
                ax.title.set_text(f'orig_images')
                ax_divider = make_axes_locatable(ax)
                cax = ax_divider.append_axes("right", size="7%", pad="2%")
                cb = fig.colorbar(im, cax=cax)

                ax = fig.add_subplot(1, 3, 2)
                im = ax.imshow(recon_images_sample, cmap='jet')
                ax.title.set_text(f'recon_images')
                ax_divider = make_axes_locatable(ax)
                cax = ax_divider.append_axes("right", size="7%", pad="2%")
                cb = fig.colorbar(im, cax=cax)

                ax = fig.add_subplot(1, 3, 3)
                im = ax.imshow(orig_images-recon_images_sample, cmap='jet')
                ax.title.set_text(f'Diff')
                ax_divider = make_axes_locatable(ax)
                cax = ax_divider.append_axes("right", size="7%", pad="2%")
                cb = fig.colorbar(im, cax=cax)
                # plt.show();raise
                plt.savefig(f'{i}.jpg')
                plt.close()


    val_loss = running_loss / counter
    return val_loss, recon_images

# def testNcompare(model, dataloader, dataset, device, criterion):
#     model.eval()
#     with torch.no_grad():
#         for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
#             # print('data shape: ', len(data))
#             data= data[0]
#             data = data.to(device)
#             reconstruction, mu, logvar = model(data)
#             data, reconstruction = data.cpu(), reconstruction.cpu()
#             # print('data, reconstruction: ', data.shape, reconstruction.shape);raise
#             # print(data.get_device(), reconstruction.get_device());raise
#             # save the last batch input and output of every epoch
#             if i == int(len(dataset)/dataloader.batch_size) - 1:
#                 orig_images = data[0,0,:,:]
#                 # print('orig_images shape: ', orig_images.shape)
#                 # plt.imshow(orig_images)
#                 # plt.show();raise
#                 recon_images = reconstruction[0,0,:,:]

#                 fig = plt.figure(figsize = (10, 3), dpi=100)
#                 plt.rcParams.update({'font.size': 15})
#                 plt.subplots_adjust(wspace=1.2, hspace = 0.5, left = 0.05, right=0.95, top=0.95, bottom=0.05)

#                 ax = fig.add_subplot(1, 3, 1)
#                 im = ax.imshow(orig_images, cmap='jet')
#                 ax.title.set_text(f'orig_images')
#                 ax_divider = make_axes_locatable(ax)
#                 cax = ax_divider.append_axes("right", size="7%", pad="2%")
#                 cb = fig.colorbar(im, cax=cax)

#                 ax = fig.add_subplot(1, 3, 2)
#                 im = ax.imshow(recon_images, cmap='jet')
#                 ax.title.set_text(f'recon_images')
#                 ax_divider = make_axes_locatable(ax)
#                 cax = ax_divider.append_axes("right", size="7%", pad="2%")
#                 cb = fig.colorbar(im, cax=cax)

#                 ax = fig.add_subplot(1, 3, 3)
#                 im = ax.imshow(orig_images-recon_images, cmap='jet')
#                 ax.title.set_text(f'Diff')
#                 ax_divider = make_axes_locatable(ax)
#                 cax = ax_divider.append_axes("right", size="7%", pad="2%")
#                 cb = fig.colorbar(im, cax=cax)
#                 # plt.show();raise
#                 plt.savefig(f'{i}.jpg')
#                 plt.close()
