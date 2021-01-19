import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class DiscriminatorNN(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class GeneratorNN(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 255),
            nn.LeakyReLU(),
            nn.Linear(255, img_dim),
            nn.Tanh(),  # should match with the Discriminator Image Input Distribution
        )

    def forward(self, x):
        return self.gen(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-3
z_dim = 64 * 4
image_dim = 28 * 28 * 1
batch_size = 64
num_epoch = 50

writer_fake = SummaryWriter(os.path.dirname(__file__) + "/logs/fake")
writer_real = SummaryWriter(os.path.dirname(__file__) + "/logs/real")

disc = DiscriminatorNN(img_dim=image_dim).to(device)
gen = GeneratorNN(z_dim=z_dim, img_dim=image_dim).to(device)

fix_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
])

dataset = datasets.MNIST(root=os.path.dirname(__file__) + "/dataset", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

criterion = nn.BCELoss()

step = 0
for epoch in range(num_epoch):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 28 * 28).to(device)
        batch_size = real.shape[0]

        # train Discriminator
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)

        disc_real = disc(real).view(-1)
        loss_d_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        loss_d_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_d = (loss_d_real + loss_d_fake) / 2
        disc.zero_grad()
        loss_d.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator
        output = disc(fake).view(-1)
        loss_g = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_g.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(f"Epoch [{epoch}/{num_epoch}] Batch {batch_idx}/{len(loader)} Loss D: {loss_d:.4f}, loss G: {loss_g:.4f}")

            with torch.no_grad():
                fake = gen(noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)

                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image("Mnist Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Mnist Real Images", img_grid_real, global_step=step)
                step += 1













