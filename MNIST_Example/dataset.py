import torch
from torchvision import datasets, transforms
import os
import shutil
import random
from PIL import Image
import numpy as np

# --- Configuration ---
DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_SAMPLES_PER_CLASS = 500
TEST_SAMPLES_PER_CLASS = 100
NUM_CLASSES = 10

def prepare_mnist_data():
    """
    Downloads the MNIST dataset and saves a subset of images into a
    custom directory structure (dataset/train/0-9 and dataset/test/0-9).
    """
    print("--- Starting MNIST Data Preparation ---")

    # 1. Download MNIST data (only for the initial file list)
    transform = transforms.ToTensor()
    try:
        train_data_raw = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_data_raw = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Error downloading MNIST: {e}")
        print("Please check your network connection.")
        return

    # 2. Organize data by class
    print("Organizing data by class...")
    train_data_by_class = [[] for _ in range(NUM_CLASSES)]
    test_data_by_class = [[] for _ in range(NUM_CLASSES)]

    for img, label in train_data_raw:
        # Convert tensor back to PIL Image for saving
        img_np = img.squeeze().numpy() * 255
        img_pil = Image.fromarray(img_np.astype(np.uint8))
        train_data_by_class[label].append(img_pil)

    for img, label in test_data_raw:
        img_np = img.squeeze().numpy() * 255
        img_pil = Image.fromarray(img_np.astype(np.uint8))
        test_data_by_class[label].append(img_pil)

    # 3. Clean up existing directories
    if os.path.exists(DATA_DIR):
        print(f"Removing existing {DATA_DIR} directory.")
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 4. Create and populate training directories
    print(f"Creating training set ({NUM_CLASSES * TRAIN_SAMPLES_PER_CLASS} total images)...")
    for i in range(NUM_CLASSES):
        class_dir = os.path.join(TRAIN_DIR, str(i))
        os.makedirs(class_dir, exist_ok=True)
        # Select random subset for training
        selected_images = random.sample(train_data_by_class[i], TRAIN_SAMPLES_PER_CLASS)

        for j, img in enumerate(selected_images):
            filepath = os.path.join(class_dir, f"{j:04d}.png")
            img.save(filepath)

    # 5. Create and populate test directories
    print(f"Creating test set ({NUM_CLASSES * TEST_SAMPLES_PER_CLASS} total images)...")
    for i in range(NUM_CLASSES):
        class_dir = os.path.join(TEST_DIR, str(i))
        os.makedirs(class_dir, exist_ok=True)
        # Select random subset for testing
        selected_images = random.sample(test_data_by_class[i], TEST_SAMPLES_PER_CLASS)

        for j, img in enumerate(selected_images):
            filepath = os.path.join(class_dir, f"{j:04d}.png")
            img.save(filepath)

    print(f"--- Data Preparation Complete. Data saved to '{DATA_DIR}/' ---")

if __name__ == "__main__":
    prepare_mnist_data()