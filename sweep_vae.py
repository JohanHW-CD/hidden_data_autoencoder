""" VAE.py only manages normal VAEs, U2_VAE.py manages multiply types,
U3_VAE.py manages multiple types with multiple different perturbation methods
SWEEP_VAE.py is U3_VAE.py when grid searching the parameters
"""

import os
import glob
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- Select mode ----------
mode = "normal"  # Options: "normal", "masked", "mask_aware"

# ---------- Paths ----------
data_path = r"C:\Users\johan\projects\data"
model_path = r"C:\Users\johan\projects\models"
plot_path = r"C:\Users\johan\projects\plots"
summary_path = os.path.join(model_path, "summary.csv")

os.makedirs(model_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

# ---------- Hyperparameters ----------
num_epochs = 500
batch_size = 64
learning_rate = 1e-3
gamma = 0.5
latent_dim = 20

# ---------- Losses ----------
def masked_loss_function(recon_x, x, M, gamma, mu, logvar):
    mse = ((recon_x - x) ** 2) * M
    mse_loss = mse.sum() / (M.sum() + 1e-8)
    KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse_loss + gamma * KL

def full_loss_function(recon_x, x, gamma, mu, logvar):
    mse_loss = F.mse_loss(recon_x, x)
    KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse_loss + gamma * KL

# ---------- VAE ----------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, x_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, x_dim),
            nn.Sigmoid()
        )

    def encode(self, xy):
        h = self.encoder(xy)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, xy):
        mu, logvar = self.encode(xy)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# ---------- Device ---------- 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Main Loop Updated for All Perturbation Methods ----------
perturbation_methods = [1, 2, 3]
modes = ["normal", "masked", "mask_aware"]
sample_sizes = [100, 400]
corruption_probs = [0.10, 0.30, 0.50]
latent_dims = [10, 30, 50]
gammas = [0.25, 0.5, 1.0]
dropouts = [0.0, 0.1]


#sample_sizes = [50, 100, 200, 400, 600, 800, 1000]
#corruption_probs = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

#latent_dims = [10, 20, 30, 40, 50]
#gammas = [0.1, 0.25, 0.5, 0.75, 1.0]
#dropouts = [0.0, 0.1, 0.3]

for latent_dim in latent_dims:
    for gamma in gammas:
        for dropout in dropouts:
            sweep_name = f"SWEEP_L{latent_dim}_G{gamma}_D{dropout}"
            sweep_data_path = os.path.join(data_path, sweep_name)
            sweep_model_path = os.path.join(model_path, sweep_name)
            sweep_plot_path = os.path.join(plot_path, sweep_name)
            sweep_summary_path = os.path.join(sweep_model_path, "summary.csv")

            os.makedirs(sweep_data_path, exist_ok=True)
            os.makedirs(sweep_model_path, exist_ok=True)
            os.makedirs(sweep_plot_path, exist_ok=True)

            for mode in modes:
                mode_short = "aware" if mode == "mask_aware" else mode
                for method in perturbation_methods:
                    for N in sample_sizes:
                        Y_clean = torch.load(os.path.join(data_path, f"Y_{N}.pt")).float()
                        for P in corruption_probs:
                            P_int = int(P * 100)
                            try:
                                X_noisy = torch.load(os.path.join(data_path, f"X_{N}_{method}_{P_int}.pt")).float()
                                M = torch.load(os.path.join(data_path, f"M_{N}_{method}_{P_int}.pt")).float()
                            except FileNotFoundError:
                                print(f"Skipping missing files for method {method}, N={N}, P={P_int}")
                                continue

                            # Prepare inputs
                            if mode == "mask_aware":
                                XY = torch.cat((X_noisy, Y_clean, M), dim=1)
                            else:
                                XY = torch.cat((X_noisy, Y_clean), dim=1)

                            x_dim = X_noisy.shape[1]
                            input_dim = XY.shape[1]

                            dataset = TensorDataset(XY, X_noisy, M)
                            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                            # Model
                            vae = VAE(input_dim=input_dim, latent_dim=latent_dim, x_dim=x_dim).to(device)
                            optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
                            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

                            best_loss = float('inf')
                            epochs_no_improve = 0
                            best_model_state = None
                            best_epoch = 0
                            loss_history = []
                            patience = 30
                            early_stop = False

                            for epoch in range(num_epochs):
                                epoch_loss = 0
                                for xy, x, m in dataloader:
                                    xy, x, m = xy.to(device), x.to(device), m.to(device)
                                    optimizer.zero_grad()
                                    recon_x, mu, logvar = vae(xy)
                                    if mode == "normal":
                                        loss = full_loss_function(recon_x, x, gamma, mu, logvar)
                                    else:
                                        loss = masked_loss_function(recon_x, x, m, gamma, mu, logvar)
                                    loss.backward()
                                    optimizer.step()
                                    epoch_loss += loss.item()

                                epoch_loss /= len(dataloader)
                                scheduler.step(epoch_loss)
                                loss_history.append(epoch_loss)

                                if epoch_loss < best_loss - 1e-6:
                                    best_loss = epoch_loss
                                    best_model_state = vae.state_dict()
                                    best_epoch = epoch
                                    epochs_no_improve = 0
                                else:
                                    epochs_no_improve += 1

                                if epochs_no_improve >= patience:
                                    early_stop = True
                                    break

                            # Reload best model
                            vae.load_state_dict(best_model_state)

                            # Save model
                            model_name = f"vae_{mode_short}_{method}_{N}_{P_int}.pth"
                            torch.save(best_model_state, os.path.join(sweep_model_path, model_name))

                            # Save reconstructed X
                            with torch.no_grad():
                                XY = XY.to(device)
                                recon_X, _, _ = vae(XY)
                            torch.save(recon_X.cpu(), os.path.join(sweep_data_path, f"X_{mode_short}_{method}_{N}_{P_int}.pt"))

                            # Plot loss
                            plt.figure()
                            plt.plot(loss_history)
                            plt.xlabel("Epoch")
                            plt.ylabel("Loss")
                            plt.title(f"VAE {mode} | Method {method} | N={N} | P={P_int}%")
                            plt.savefig(os.path.join(sweep_plot_path, f"loss_{mode_short}_{method}_{N}_{P_int}.png"))
                            plt.close()

                            # CSV log
                            if os.path.exists(sweep_summary_path):
                                df = pd.read_csv(summary_path)
                            else:
                                df = pd.DataFrame(columns=["Mode", "Method", "N", "P", "Final Loss", "Model Name"])

                            # Remove any existing entry with same (Mode, Method, N, P)
                            df = df[~((df['Mode'] == mode_short) & (df['Method'] == method) & (df['N'] == N) & (df['P'] == P_int))]

                            # Build a well-formed new row
                            new_row = pd.DataFrame([{
                                "Mode": str(mode_short),
                                "Method": str(method),
                                "N": int(N),
                                "P": int(P_int),
                                "Final Loss": float(loss_history[-1]) if loss_history else float('nan'),
                                "Model Name": str(model_name)
                            }])

                            # Concatenate cleanly
                            df = pd.concat([df, new_row], ignore_index=True)

                            df.to_csv(summary_path, index=False)

                            print(f"Trained VAE ({mode}, method {method}) for N={N}, P={P_int}%, with Dim={latent_dim}, Gamma={gamma}, Dropout={dropout} | Best loss: {best_loss:.4f} | Epoch {best_epoch+1}")

print("All VAE models trained and saved.")
