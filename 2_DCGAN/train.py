import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

IN_CHANNELS = 1
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_D, FEATURES_G = 64, 64
LR = 2e-4
IMAGE_SIZE = 64
BATCH_SIZE = 128
NUM_WORKERS = 0

transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IN_CHANNELS)], [0.5 for _ in range(IN_CHANNELS)]),
])

dataset = datasets.MNIST(root='./datasets', train=True, transform=transforms, download=True)
loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

model_d = Discriminator(IN_CHANNELS, FEATURES_D).to(device)
initialize_weights(model_d)
model_g = Generator(Z_DIM, IN_CHANNELS, FEATURES_G).to(device)
initialize_weights(model_g)
criterion = nn.BCELoss()
optim_gen = torch.optim.Adam(model_g.parameters(), LR, betas=(0.5, 0.999))
optim_disc = torch.optim.Adam(model_d.parameters(), LR, betas=(0.5, 0.999))

writer_real = SummaryWriter(log_dir='logs/real')
writer_fake = SummaryWriter(log_dir='logs/fake')
step = 0

FIXED_NOISE = torch.randn((32, Z_DIM, 1, 1)).to(device)

model_d.train()
model_g.train()

for epoch in tqdm(range(NUM_EPOCHS)):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = model_g(noise)

        disc_d = model_d(real).reshape(-1)
        loss_disc_d = criterion(disc_d, torch.ones_like(disc_d))
        disc_g = model_d(fake).reshape(-1)
        loss_disc_g = criterion(disc_g, torch.zeros_like(disc_g))
        loss_disc = (loss_disc_d + loss_disc_g) / 2
        model_d.zero_grad()
        loss_disc.backward(retain_graph=True)
        optim_disc.step()

        gen_g = model_d(fake).reshape(-1)
        loss_gen = criterion(gen_g, torch.ones_like(gen_g))
        
        model_g.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch: {epoch+1}/{NUM_EPOCHS} || Batch: {batch_idx}/{len(loader)} || Loss D: {loss_disc:.4f} || Loss G: {loss_gen:.4f}"
            ) 

        with torch.inference_mode():
            fake = model_g(FIXED_NOISE)
            # take out 32 examples
            img_grid_real = torchvision.utils.make_grid(
                real[:32], normalize=True
            )
            img_grid_fake = torchvision.utils.make_grid(
                fake[:32], normalize=True
            )

            writer_real.add_image("Real", img_grid_real, global_step=step)
            writer_fake.add_image("Fake", img_grid_fake, global_step=step)
        
        step+=1