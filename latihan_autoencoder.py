import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()


        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )


    def forward(self,x):
        x = x.view(-1, 28*28)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, 1, 28, 28)
    


train_loader = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("started Training")
for epoch in range(10):
    for images,_ in train_loader:
        images = images.to(device)

        reconstructed = model(images)
        loss = criterion(reconstructed,images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"current epoch is {epoch}, with loss is {loss.item()}")

print("Anomaly detection")
torch.save(model.state_dict(), "Custome_AE.pth")

test_tensor = images[0]
recon_tensor = reconstructed[0]

# 2. Calculate the difference (Anomaly Map)
# We do this on Tensors before converting to Numpy
diff_tensor = torch.abs(test_tensor - recon_tensor)

# 3. Convert all to Numpy for Plotting
test_img = test_tensor.detach().cpu().squeeze().numpy()
reconstructed_img = recon_tensor.detach().cpu().squeeze().numpy()
difference_img = diff_tensor.detach().cpu().squeeze().numpy()


print("--------------Matplotlib-----------")

# -------------- Matplotlib -----------
plt.figure(figsize=(15, 5)) # Made wider for 3 images

# Plot Original
plt.subplot(1, 3, 1) # (Rows, Cols, Index)
plt.title("Original Image")
plt.imshow(test_img, cmap="gray")

# Plot Reconstruction
plt.subplot(1, 3, 2)
plt.title("AI Reconstruction")
plt.imshow(reconstructed_img, cmap='gray')

# Plot Difference (The Anomaly Map)
plt.subplot(1, 3, 3)
plt.title("Difference Map")
# Using 'hot' cmap makes anomalies glow red/yellow!
plt.imshow(difference_img, cmap='hot') 

plt.show()