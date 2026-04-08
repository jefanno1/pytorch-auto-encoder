import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),  # Fixed: Capital U
            nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),  # Fixed: Capital U
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, 1, 28, 28)

# Fixed: Added missing comma after ToTensor()
train_loader = DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor()), 
    batch_size=32, 
    shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Starting training...")
for epoch in range(5): # 5 epochs is enough for MNIST to learn
    for images, _ in train_loader:
        images = images.to(device)

        reconstructed = model(images)
        loss = criterion(reconstructed, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} complete. Loss: {loss.item():.4f}")

# --- ANOMALY DETECTION TEST ---
print("\nTesting Anomaly Detection...")
# Let's take one image from the last batch
test_img = images[0] 
recon_img = reconstructed[0]

# Fixed: use 'images' instead of 'original_image'
diff = torch.abs(test_img - recon_img)

print(f"Max difference in pixels: {diff.max().item():.4f}")

if diff.max() > 0.5:
    print("ANOMALY DETECTED: Unknown pattern found!")
else:
    print("Pattern looks normal.")

original = images[0].detach().cpu().squeeze().numpy()
reconstruction = reconstructed[0].detach().cpu().squeeze().numpy()

# 2. Create a side-by-side plot
plt.figure(figsize=(10, 5))

# Plot Original
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original, cmap='gray')

# Plot Reconstruction
plt.subplot(1, 2, 2)
plt.title("AI Reconstruction")
plt.imshow(reconstruction, cmap='gray')

plt.show()