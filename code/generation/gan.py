import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import *
from options import Options
import sys
from utils import *
from dataset import create_power_dataloader
import matplotlib.pyplot as plt
import pickle as pkl
import os

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        # Map noise to sequence
        self.fc_layers = nn.Sequential(
            nn.Linear(opt.latent_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, opt.seq_len),
            nn.Sigmoid()  # Output values between 0-1
        )

    def forward(self, noise):
        # noise: (batch_size, latent_size)
        x = self.fc_layers(noise)
        return x.unsqueeze(-1)  # (batch_size, seq_len, 1)

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.opt = opt
        # Convolutional layers extract features
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        
        # Calculate feature size after convolution
        conv_output_size = opt.seq_len // 8 * 256
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, seq_len, 1)
        x = x.transpose(1, 2)  # (batch_size, 1, seq_len)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc_layers(x).squeeze()

class GAN:
    def __init__(self, opt, dataloader):
        super(GAN, self).__init__()
        self.opt = opt
        self.dataloader = dataloader
        self.netG = Generator(opt).to(opt.device)
        self.netD = Discriminator(opt).to(opt.device)
        self.adversarial_loss_func = nn.BCELoss()
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_G, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(0.5, 0.999))

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.opt.latent_size, device=self.opt.device)

    def forward_G(self):
        self.noise = self.sample_noise(self.batch_size)
        self.x_gen = self.netG(self.noise)
        self.y_gen = self.netD(self.x_gen)

    def train(self):
        netG_loss_set, netD_loss_set = [], []
        for epoch in range(self.opt.n_epochs):
            for i, data in enumerate(self.dataloader):
                # Prepare data
                self.x_real = data["power"].to(self.opt.device)
                self.batch_size = self.x_real.shape[0]
                
                # Real and fake labels
                self.valid = torch.ones(self.batch_size, dtype=torch.float32).to(self.opt.device)
                self.fake = torch.zeros(self.batch_size, dtype=torch.float32).to(self.opt.device)
                
                # Train discriminator
                self.optimizer_D.zero_grad()
                
                # Real data
                self.y_real = self.netD(self.x_real)
                self.real_loss = self.adversarial_loss_func(self.y_real, self.valid)
                
                # Generate fake data
                self.noise = self.sample_noise(self.batch_size)
                self.x_gen = self.netG(self.noise)
                self.y_gen = self.netD(self.x_gen.detach())
                self.fake_loss = self.adversarial_loss_func(self.y_gen, self.fake)
                
                self.netD_loss = 0.5 * (self.real_loss + self.fake_loss)
                self.netD_loss.backward()
                self.optimizer_D.step()
                netD_loss_set.append(self.netD_loss.item())
                
                # Train generator
                self.optimizer_G.zero_grad()
                self.y_gen = self.netD(self.x_gen)
                self.netG_loss = self.adversarial_loss_func(self.y_gen, self.valid)
                self.netG_loss.backward()
                self.optimizer_G.step()
                netG_loss_set.append(self.netG_loss.item())
                
                if i % 10 == 0:  # Reduce print frequency
                    print(f"epoch={epoch}/{self.opt.n_epochs}, iteration={i}, "
                          f"netG_loss={netG_loss_set[-1]:.4f}, netD_loss={netD_loss_set[-1]:.4f}")
            
            # Save model
            save_dir = f"weights/{self.opt.model_name}/{self.opt.level}"
            os.makedirs(save_dir, exist_ok=True)
            model_para = {"G": self.netG.state_dict(), "D": self.netD.state_dict()}
            save_path = f"{save_dir}/epoch{epoch}.pt"
            torch.save(model_para, save_path)
            
        plot_labels = ["Generator Loss", "Discriminator Loss"]
        plot_training_loss(netG_loss_set, netD_loss_set, model_name=self.opt.model_name, labels=plot_labels)

    def test(self, weight_path, sample_num, output_dir="generation/gan"):
        """Generate samples and save"""
        with torch.no_grad():
            netG_weight = torch.load(weight_path, map_location=self.opt.device)["G"]
            self.netG.load_state_dict(netG_weight)
            self.netG.eval()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        noise = self.sample_noise(sample_num)
        x_gen = self.netG(noise)
        generated_samples = []
        
        for i in range(sample_num):
            x = x_gen[i].detach().cpu().numpy().squeeze()
            
            # Ensure values are positive
            x = np.maximum(x, 0)
            # Ensure values are within reasonable range
            x = np.minimum(x, 1)
            generated_samples.append(x)
            
            # Save individual sample image
            plt.figure(figsize=(12, 4))
            plt.plot(x)
            plt.title(f'GAN Generated Power Curve #{i}')
            plt.xlabel('Time Steps (15min intervals)')
            plt.ylabel('Normalized Power')
            plt.grid(True)
            plt.savefig(f"{output_dir}/sample_{i}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save as pickle file
            with open(f"{output_dir}/sample_{i}.pkl", "wb") as f:
                pkl.dump(x, f)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{sample_num} samples")
        
        # Save all generated samples
        np.save(f"{output_dir}/generated_samples.npy", np.array(generated_samples))
        print(f"All {sample_num} samples saved to {output_dir}")
        
        return generated_samples

if __name__ == "__main__":
    isTrain = True  # Set to True for training
    model_name = "gan"
    opt = Options(model_name, isTrain)
    
    # Use charging station power data
    data_loader, dataset = create_power_dataloader("A1.csv", opt.batch_size, opt.shuffle)
    
    model = GAN(opt, data_loader)
    
    if isTrain:
        model.train()
    else:
        best_epoch = 30
        pt_file = f"weights/gan/{opt.level}/epoch{best_epoch}.pt"
        model.test(pt_file, sample_num=100)