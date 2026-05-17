import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from cnn_network import ConvAutoencoder

def train_model(epochs=10, batch_size=64, learning_rate=1e-3, save_dir="saved_models"):
    print("Starting Training for ConvAutoencoder...")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'cnn_autoencoder.pth')

    # Data transformation
    # Convert PIL Image to tensor (values between 0 and 1)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load CIFAR-10 dataset (general natural images)
    print("Loading CIFAR-10 dataset...")
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)

    # Initialize model, loss, and optimizer
    model = ConvAutoencoder(in_channels=3, latent_channels=16).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Training started...")
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            # Forward pass (Autoencoder tries to reconstruct its input)
            outputs = model(images)
            loss = criterion(outputs, images)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Save the model weights
    print(f"Saving model to {model_path}...")
    torch.save(model.state_dict(), model_path)
    print("Training complete!")

if __name__ == '__main__':
    # For proof of concept, 5 epochs is enough to learn basic reconstruction.
    train_model(epochs=5, batch_size=128)
