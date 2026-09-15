import torch.nn as nn
import torch
import itertools
from options import Options
import sys
from utils import *
from dataset import create_power_dataloader
import matplotlib.pyplot as plt
import pickle as pkl
import os
import numpy as np

def reparameterization(mu, std, opt):
    sampled_z = torch.randn(mu.size(0), opt.latent_size, device=opt.device)
    z = sampled_z * std + mu
    return z

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.encoder = nn.Sequential(
            nn.Linear(opt.seq_len, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
        )
        self.mu = nn.Linear(512, opt.latent_size)
        self.std = nn.Linear(512, opt.latent_size)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        std = self.std(x)
        z = reparameterization(mu, std, self.opt)
        return z, x

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(opt.latent_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, opt.seq_len),
            nn.ReLU()
        )

    def forward(self, z):
        x = self.decoder(z)
        return x

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.netD = nn.Sequential(
            nn.Linear(opt.latent_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        y = self.netD(z)
        return y

class AAE:
    def __init__(self, opt, dataloader):
        super(AAE, self).__init__()
        self.opt = opt
        self.dataloader = dataloader
        self.encoder = Encoder(opt).to(opt.device)
        self.decoder = Decoder(opt).to(opt.device)
        self.netD = Discriminator(opt).to(opt.device)
        self.rec_loss_func = nn.L1Loss()
        self.enc_loss_func = nn.MSELoss()
        self.adversarial_loss = torch.nn.BCELoss()
        self.optimizer_AE = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
                                             lr=opt.lr_AE, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(0.5, 0.999))

    def forward_AE(self):
        self.z_real, self.h_real = self.encoder(self.x_real)
        self.x_rec = self.decoder(self.z_real)
        self.z_rec, self.h_rec = self.encoder(self.x_rec)
        self.input_x, self.target_x = self.x_rec[:, 0:-1], self.x_rec[:, 1:]

    def step_AE(self):
        self.forward_AE()
        self.rec_loss = self.rec_loss_func(self.x_real, self.x_rec)
        self.enc_loss = self.enc_loss_func(self.h_real, self.h_rec)
        self.gen_loss = self.adversarial_loss(self.netD(self.z_real), self.valid)
        self.ae_loss = self.opt.alpha*self.rec_loss + (1-self.opt.alpha)*self.gen_loss + self.opt.beta*self.enc_loss
        self.optimizer_AE.zero_grad()
        self.ae_loss.backward()
        self.optimizer_AE.step()

    def step_D(self, z):
        real_loss = self.adversarial_loss(self.netD(z), self.valid)
        fake_loss = self.adversarial_loss(self.netD(self.z_real.detach()), self.fake)
        self.adv_loss = 0.5 * (real_loss + fake_loss)
        self.optimizer_D.zero_grad()
        self.adv_loss.backward()
        self.optimizer_D.step()

    def train(self):
        epoch_rec_loss, epoch_enc_loss, epoch_G_loss, epoch_D_loss = [], [], [], []
        for epoch in range(self.opt.n_epochs):
            batch_rec_loss, batch_enc_loss, batch_G_loss, batch_D_loss = [], [], [], []
            for i, data in enumerate(self.dataloader):
                # Charging station power data
                self.x_real = data["power"].to(self.opt.device).squeeze(2)
                
                batch_size = self.x_real.shape[0]
                self.valid = torch.ones((batch_size, 1), dtype=torch.float32).to(self.opt.device)
                self.fake = torch.zeros((batch_size, 1), dtype=torch.float32).to(self.opt.device)
                z = torch.randn(batch_size, self.opt.latent_size, device=self.opt.device)
                
                self.step_AE()
                self.step_D(z)
                
                batch_rec_loss.append(self.rec_loss.item())
                batch_enc_loss.append(self.enc_loss.item())
                batch_G_loss.append(self.gen_loss.item())
                batch_D_loss.append(self.adv_loss.item())
            
            epoch_rec_loss.append(np.mean(batch_rec_loss))
            epoch_enc_loss.append(np.mean(batch_enc_loss))
            epoch_G_loss.append(np.mean(batch_G_loss))
            epoch_D_loss.append(np.mean(batch_D_loss))
            
            print(f"epoch={epoch}/{self.opt.n_epochs}, rec_loss={epoch_rec_loss[-1]:.4f}, "
                  f"enc_loss={epoch_enc_loss[-1]:.4f}, G_loss={epoch_G_loss[-1]:.4f}, "
                  f"D_loss={epoch_D_loss[-1]:.4f}")
            
            # Save model
            save_dir = f"weights/{self.opt.model_name}/{self.opt.level}"
            os.makedirs(save_dir, exist_ok=True)
            model_para = {"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict()}
            save_path = f"{save_dir}/epoch{epoch}.pt"
            torch.save(model_para, save_path)
            
        plot_training_loss(epoch_rec_loss, epoch_enc_loss, epoch_G_loss, epoch_D_loss,
                           model_name=f"{self.opt.model_name}",
                           labels=["Reconstruction Loss", "Encoding Loss", "G Loss", "D Loss"])

    def test(self, pt_file, sample_num, output_dir="generation/vae"):
        """Generate samples and save"""
        weight = torch.load(pt_file, map_location=self.opt.device)
        self.encoder.load_state_dict(weight["encoder"])
        self.decoder.load_state_dict(weight["decoder"])
        self.encoder.eval()
        self.decoder.eval()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        z = torch.randn(sample_num, self.opt.latent_size, device=self.opt.device)
        x_gen = self.decoder(z).squeeze().detach().cpu().numpy()
        generated_samples = []
        
        for i in range(sample_num):
            if x_gen.ndim == 1:  # Only one sample
                x = x_gen
            else:
                x = x_gen[i]
            
            # Ensure values are positive
            x = np.maximum(x, 0)
            generated_samples.append(x)
            
            # Save individual sample image
            plt.figure(figsize=(12, 4))
            plt.plot(x)
            plt.title(f'VAE Generated Power Curve #{i}')
            plt.xlabel('Time Steps (15min intervals)')
            plt.ylabel('Normalized Power')
            plt.grid(True)
            plt.savefig(f"{output_dir}/sample_{i}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save as pickle file
            with open(f"{output_dir}/sample_{i}.pkl", "wb") as f:
                pkl.dump(x, f)
            
            if sample_num == 1:  # Break loop when only one sample
                break
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{sample_num} samples")
        
        # Save all generated samples
        np.save(f"{output_dir}/generated_samples.npy", np.array(generated_samples))
        print(f"All {len(generated_samples)} samples saved to {output_dir}")
        
        return generated_samples

if __name__ == "__main__":
    isTrain = True  # Set to True for training
    model_name = "aae"
    opt = Options(model_name, isTrain)
    
    # Use charging station power data
    data_loader, dataset = create_power_dataloader("A1.csv", opt.batch_size, opt.shuffle)
    
    model = AAE(opt, data_loader)
    
    if isTrain:
        model.train()
    else:
        best_epoch = 152
        pt_file = f"weights/aae/{opt.level}/epoch{best_epoch}.pt"
        model.test(pt_file, sample_num=100)
