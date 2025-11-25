import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import sys

# Ensure parent directory is in path to import dataset/model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Attempting relative imports which are safer for intra-directory structure
try:
    from .dataset import prepare_mnist_data, DATA_DIR 
    from .model import SimpleDenseNet
except ImportError:
    # Fallback if running main.py directly
    from dataset import prepare_mnist_data, DATA_DIR 
    from model import SimpleDenseNet

# --- Configuration ---
BATCH_SIZE = 16 # Increased batch size slightly for better GPU utilization
LEARNING_RATE = 0.001
EPOCHS = 10 # Changed back to 5 for a quick baseline, feel free to increase this
MODEL_PATH = "mnist_model_fp32_baseline.pth" # Renamed to reflect FP32 baseline

# Data transforms for preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Ensure images are single channel
    transforms.Resize((28, 28)),                
    transforms.ToTensor(),                      # Convert PIL Image to Tensor (float32, [0, 1])
    transforms.Normalize((0.5,), (0.5,))        # Normalize to [-1, 1]
])

def load_data():
    """Loads dataset from the structured directories."""
    
    # Check if data directory exists, if not, prepare it.
    if not os.path.exists(os.path.join(DATA_DIR, "train")):
        prepare_mnist_data()

    # PyTorch's ImageFolder handles the directory structure (class names are subdirectories)
    try:
        train_dataset = ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=transform)
        test_dataset = ImageFolder(root=os.path.join(DATA_DIR, "test"), transform=transform)
    except FileNotFoundError:
        print(f"Error: Could not find data in {DATA_DIR}. Please run dataset.py standalone first or check path.")
        sys.exit(1)

    # Use pin_memory=True for faster GPU data transfer
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, device):
    """Handles the training loop for the model."""
    model.train()
    
    # Using EPOCHS from main scope for printout consistency
    total_epochs = EPOCHS
    print(f"\nStarting stable training in FP32 for {total_epochs} epochs...")
    
    for epoch in range(total_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            
            # --- FIX: Images remain torch.float32 (default) ---
            images = images.to(device)                      # Move to CUDA (as FP32)
            images = images.view(images.size(0), -1)        # Flatten to (batch_size, 784)
            labels = labels.to(device)                      # Labels remain INT64

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if (i + 1) % 50 == 0: # Check every 50 steps for better feedback
                print(f"Epoch [{epoch+1}/{total_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/50:.4f}")
                running_loss = 0.0
        
        # Evaluate after each epoch
        acc = evaluate_model(model, test_loader_global, device, verbose=False)
        print(f"Epoch {epoch+1} completed. Test Accuracy: {acc:.2f}%")
        
def evaluate_model(model, test_loader, device, verbose=True):
    """Evaluates the model on the test set and returns accuracy."""
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad(): # Disable gradient calculation
        for images, labels in test_loader:
            
            # --- FIX: Images remain torch.float32 (default) ---
            images = images.to(device) # Move to CUDA (as FP32)
            images = images.view(images.size(0), -1)        # Flatten
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    if verbose:
        print(f"\nAccuracy on the 1000 test images: {accuracy:.2f}%")
    return accuracy

# Global variable to hold the test loader for evaluation inside the training loop
test_loader_global = None

def main():
    global test_loader_global
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    train_loader, test_loader = load_data()
    test_loader_global = test_loader # Set global for inside-train evaluation

    # 3. Initialize Model, Loss, and Optimizer
    model = SimpleDenseNet().to(device)
    
    # --- CRITICAL FIX: Removed model.half() for stable FP32 training ---
    # The model now trains in the default torch.float32 precision.
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Train
    train_model(model, train_loader, criterion, optimizer, device)

    # 5. Evaluate (Final)
    evaluate_model(model, test_loader, device)
    
    # 6. Save the final model (as the FP32 baseline)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nFinal FP32 baseline model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()