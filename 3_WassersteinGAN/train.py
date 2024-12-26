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

IN_CHANNELS = 3
Z_DIM = 100
NUM_EPOCHS = 15
FEATURES_D, FEATURES_G = 64, 64
LR = 5e-5
IMAGE_SIZE = 64
BATCH_SIZE = 64
NUM_WORKERS = 0
CRITIC_ITERATIONS = 5
WEIGHT_CLIPS = 0.01

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(IN_CHANNELS)], [0.5 for _ in range(IN_CHANNELS)]),
])

# dataset = datasets.MNIST(root='./datasets', train=True, transform=transform, download=True)
dataset = datasets.ImageFolder(root=r'C:\Users\yniti\OneDrive\Desktop\Practice Practice Practice\datasets\celeb_dataset', transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

critic = Discriminator(IN_CHANNELS, FEATURES_D).to(device)
initialize_weights(critic)
model_g = Generator(Z_DIM, IN_CHANNELS, FEATURES_G).to(device)
initialize_weights(model_g)
optim_gen = torch.optim.RMSprop(model_g.parameters(), lr=LR)
optim_critic = torch.optim.RMSprop(critic.parameters(), lr=LR)

writer_real = SummaryWriter(log_dir='logs/real')
writer_fake = SummaryWriter(log_dir='logs/fake')
step = 0
FIXED_NOISE = torch.randn((32, Z_DIM, 1, 1)).to(device)

critic.train()
model_g.train()

for epoch in tqdm(range(NUM_EPOCHS)):
    # Turn loader into an iterator so we can fetch multiple batches per generator iteration
    loader_iter = iter(loader)

    while True:
        # Perform multiple critic iterations
        for _ in range(CRITIC_ITERATIONS):
            try:
                real, _ = next(loader_iter)
            except StopIteration:
                # If we run out of data during the critic updates, end this epoch
                break

            real = real.to(device)
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = model_g(noise)
            
            # Compute critic scores
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake))

            critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            optim_critic.step()

            # Weight clipping for WGAN
            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIPS, WEIGHT_CLIPS)
        else:
            # If we completed all CRITIC_ITERATIONS without StopIteration, proceed to generator step.
            # Update generator
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = model_g(noise)
            output = critic(fake).reshape(-1)
            gen_loss = -torch.mean(output)

            model_g.zero_grad()
            gen_loss.backward()
            optim_gen.step()

            if step % 100 == 0:
                print(
                    f"Epoch: {epoch+1}/{NUM_EPOCHS} || Step: {step} || Loss D: {critic_loss:.4f} || Loss G: {gen_loss:.4f}"
                )

            with torch.inference_mode():
                fake = model_g(FIXED_NOISE)
                # take out 32 examples from the last real batch we used
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
            continue  # Move to next set of critic iterations

        # If we hit StopIteration in the critic loop, break out of the while loop to start a new epoch
        break
