import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# -------------------------------
# VAE Model Definition
# -------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim=5, latent_dim=2):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc21 = nn.Linear(16, latent_dim)
        self.fc22 = nn.Linear(16, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 16)
        self.fc4 = nn.Linear(16, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# -------------------------------
# Loss Function
# -------------------------------
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# -------------------------------
# Load and Prepare Dataset
# -------------------------------
df = pd.read_csv("static/documents.csv")

# Separate features and labels
X = df.drop("label", axis=1).values
y = df["label"].values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.int64)

# Use only normal (label=0) data for training
X_train = X_tensor[y_tensor == 0]

# -------------------------------
# Train the VAE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(input_dim=X_train.shape[1]).to(device)
optimizer = optim.Adam(vae.parameters(), lr=0.001)
X_train = X_train.to(device)

# Training loop
for epoch in range(50):
    vae.train()
    optimizer.zero_grad()
    recon, mu, logvar = vae(X_train)
    loss = loss_function(recon, X_train, mu, logvar)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(vae.state_dict(), "models/vae_attack_model.pt")

# -------------------------------
# Evaluate on Full Data
# -------------------------------
vae.eval()
X_tensor = X_tensor.to(device)

with torch.no_grad():
    recon_all, _, _ = vae(X_tensor)
    recon_error = torch.mean((X_tensor - recon_all)**2, dim=1).cpu().numpy()

# Threshold: 95th percentile of normal errors
threshold = np.percentile(recon_error[y == 0], 95)
y_pred = (recon_error > threshold).astype(int)

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y, y_pred))

# Optional: Plot Histogram
plt.hist(recon_error, bins=50)
plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.4f}")
plt.title("Reconstruction Error Histogram")
plt.xlabel("Reconstruction Error")
plt.ylabel("Document Count")
plt.legend()
plt.grid(True)
plt.show()
