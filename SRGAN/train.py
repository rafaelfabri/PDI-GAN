import torch
import config
import torchvision
from torch import nn 
from torch import optim
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision import datasets
from utils import save_checkpoint
from dataset import MyImageFolder
import numpy as np 
from PIL import Image
from utils import plot_examples


torch.backends.cudnn.benchmark = True

def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss, last_epoch, epoch):
    loop = tqdm(loader, leave=True)
    i = 0 
    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        #discriminator
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        #generator
        disc_fake = disc(fake)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
        gen_loss = loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        i = i + 1
        if last_epoch == (epoch + 1):
            print(i)
            plot_examples("/home/rafaelfabrichimidt/Documentos/projetos/Mestrado/PDI/SRGAN/fake", gen)
            #print(fake[0][0][:][:])
            #fake_np = fake[0][0][:][:].detach().cpu().numpy()
            #print(fake_np)
            #image = Image.fromarray(np.uint8(fake_np))
            ##image.save(f"SRGAN/fake/{i}.png")
            #print('fake', low_res)
            #print('fake', fake)

def main():

    #path_train = '/home/rafaelfabrichimidt/Documentos/projetos/Mestrado/PDI/artigo/Images/train'

    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((48,48)),
    #     v2.RandomHorizontalFlip(p=0.5),
    #     #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     torchvision.transforms.ToTensor()
    # ])

    dataset = MyImageFolder(root_dir="/home/rafaelfabrichimidt/Documentos/projetos/Mestrado/PDI/artigo/Images/train__")
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    gen = Generator(in_channels=1).to(config.DEVICE)
    disc = Discriminator(in_channels=1).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()
    
    
    for epoch in range(config.NUM_EPOCHS):
        train_fn(dataloader, 
                 disc=disc, 
                 gen=gen, 
                 opt_gen=opt_gen,
                 opt_disc=opt_disc,
                 mse=mse,
                 bce=bce,
                 vgg_loss=vgg_loss, 
                 last_epoch=config.NUM_EPOCHS,
                 epoch=epoch
        )
        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()