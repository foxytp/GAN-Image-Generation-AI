import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image

# Main configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
img_size = 64
channels = 3  # 3 for color, 1 for grayscale
batch_size = 64
epochs = 10000
sample_interval = 100
checkpoint_interval = 500
checkpoint_path = "training_checkpoints/latest_checkpoint.pth"

# Create necessary directories
os.makedirs("generated_images", exist_ok=True)
os.makedirs("training_checkpoints", exist_ok=True)
os.makedirs("meta", exist_ok=True)

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, f) for f in os.listdir(root)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # Dummy label

# Transformations with data augmentation
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), channels, img_size, img_size)
        return img

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(channels * img_size * img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        flattened = img.view(img.size(0), -1)
        return self.model(flattened)

# Initialize models and optimizers
generator = Generator().to(device)
discriminator = Discriminator().to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
adversarial_loss = nn.BCELoss()

# Load dataset
dataset = CustomDataset(root="./meta", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Function to load checkpoint
def load_checkpoint():
    if os.path.exists(checkpoint_path):
        print("\nLoading last checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        return checkpoint['epoch'] + 1
    return 0

# Improved training function
def train_gan():
    start_epoch = load_checkpoint()

    for epoch in range(start_epoch, epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # Train Discriminator
            optimizer_D.zero_grad()

            # Real images
            real_loss = adversarial_loss(discriminator(real_imgs),
                                         torch.ones(batch_size, 1).to(device))

            # Generated images
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            fake_loss = adversarial_loss(discriminator(fake_imgs.detach()),
                                         torch.zeros(batch_size, 1).to(device))

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            g_loss = adversarial_loss(discriminator(fake_imgs),
                                      torch.ones(batch_size, 1).to(device))
            g_loss.backward()
            optimizer_G.step()

        # Save progress
        if epoch % sample_interval == 0:
            print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
            save_image(fake_imgs.data[:25],
                       f"generated_images/epoch_{epoch}.png",
                       nrow=5, normalize=True)

        # Save checkpoint
        if epoch % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
            }, checkpoint_path)

    print("Training completed!")

if __name__ == "__main__":
    train_gan()

