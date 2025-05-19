import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import subprocess

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
image_size = 64    # resize images to this size
nc = 3            # number of channels (e.g., 1 for grayscale, 3 for RGB)
z_dim = 100       # latent vector size
ngf = 64          # generator feature maps
df = 64           # discriminator feature maps
num_epochs = 200   # max epochs
early_stop_patience = num_epochs  # epochs without FID improvement
lr = 0.0002
beta1 = 0.5       # beta1 for Adam optimizer

# Paths for train and validation folders
label = 4
print(f'Label: {label}')
train_dir = f'/home/thiago/Downloads/train/images/{label}/train_OBJ'
val_dir = f'/home/thiago/Downloads/test_separados/{label}'

# Create output directories
os.makedirs(f"./output_{label}/images/grids", exist_ok=True)
os.makedirs(f"./output_{label}/images/samples", exist_ok=True)

# Transformations for both train and val
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*nc, [0.5]*nc)
])

# Load datasets
dataset_train = datasets.ImageFolder(root=train_dir, transform=transform)
dataset_val = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)

# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator architecture
generator = nn.Sequential(
    nn.ConvTranspose2d(z_dim, ngf*8, 4, 1, 0, bias=False),
    nn.BatchNorm2d(ngf*8), nn.ReLU(True),
    nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf*4), nn.ReLU(True),
    nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf*2), nn.ReLU(True),
    nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf), nn.ReLU(True),
    nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
    nn.Tanh()
).to(device)

# Discriminator architecture
discriminator = nn.Sequential(
    nn.Conv2d(nc, df, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(df, df*2, 4, 2, 1, bias=False), nn.BatchNorm2d(df*2), nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(df*2, df*4, 4, 2, 1, bias=False), nn.BatchNorm2d(df*4), nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(df*4, df*8, 4, 2, 1, bias=False), nn.BatchNorm2d(df*8), nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(df*8, 1, 4, 1, 0, bias=False), nn.Sigmoid()
).to(device)

# Apply weight initialization
generator.apply(weights_init)
discriminator.apply(weights_init)

# Loss and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

# Fixed noise for consistent sampling
fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

# Early stopping variables
best_fid = float('inf')
epochs_no_improve = 0

print("Starting Training...")
for epoch in range(1, num_epochs+1):
    for i, (real_images, _) in enumerate(train_loader, 1):
        b_size = real_images.size(0)
        real_images = real_images.to(device)

        # Real and fake labels
        real_labels = torch.full((b_size,), 1., device=device)
        fake_labels = torch.full((b_size,), 0., device=device)

        # Train Discriminator
        discriminator.zero_grad()
        output_real = discriminator(real_images).view(-1)
        lossD_real = criterion(output_real, real_labels)
        noise = torch.randn(b_size, z_dim, 1, 1, device=device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach()).view(-1)
        lossD_fake = criterion(output_fake, fake_labels)
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # Train Generator
        generator.zero_grad()
        output = discriminator(fake_images).view(-1)
        lossG = criterion(output, real_labels)
        lossG.backward()
        optimizerG.step()

    # Save grid of images for visual inspection
    with torch.no_grad():
        fake_grid = generator(fixed_noise).cpu()
        utils.save_image(fake_grid, f"./output_{label}/images/grids/fake_epoch_{epoch}.png", normalize=True)

    # Save individual images for FID calculation
    with torch.no_grad():
        sample_noise = torch.randn(len(dataset_val), z_dim, 1, 1, device=device)
        fake_samples = generator(sample_noise).cpu()
        sample_dir = f"./output_{label}/images/samples/epoch_{epoch}"
        os.makedirs(sample_dir, exist_ok=True)
        for i, img in enumerate(fake_samples):
            utils.save_image(img, f"{sample_dir}/fake_{i:04d}.png", normalize=True)

    # Calculate FID
    # print(sample_dir)
    # print(f'{val_dir}/img')
    cmd = ['pytorch-fid', sample_dir, f'{val_dir}/img', '--device', 'cuda']
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        fid_value = float(result.stdout.strip().split()[-1])
    except:
        fid_value = None

    print(f"Epoch {epoch} - FID: {fid_value}")

    # Early stopping
    if fid_value is not None and fid_value < best_fid:
        best_fid = fid_value
        epochs_no_improve = 0
        torch.save(generator.state_dict(), f"./output_{label}/best_generator.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print(f"No FID improvement for {early_stop_patience} epochs. Stopping.")
            break

# Save final models
torch.save(generator.state_dict(), f"./output_{label}/final_generator.pth")
torch.save(discriminator.state_dict(), f"./output_{label}/final_discriminator.pth")
print("Training finished. Best FID:", best_fid)