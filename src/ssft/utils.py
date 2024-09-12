import os
import psutil
import torch
from timm import create_model
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import gc


# Function to calculate the average image file size in the folder and subfolders
def calculate_average_image_size(folder):
    file_sizes = []
    # Walk through the folder and its subdirectories
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(root, filename)
                file_sizes.append(os.path.getsize(file_path))
    
    avg_size = np.mean(file_sizes) if file_sizes else 0
    return avg_size


# Function to get available system RAM
def get_available_ram():
    mem = psutil.virtual_memory()
    return mem.available


# Function to get the number of CPU cores
def get_cpu_cores():
    return psutil.cpu_count(logical=False)


# Function to estimate memory per worker including augmentation and target transform
def estimate_worker_memory_usage(avg_image_size, batch_size, model_param_size, aug_complexity_factor, target_complexity_factor):
    # Estimate how much memory each worker will need
    image_memory_usage = avg_image_size * batch_size
    
    # Adjust memory usage by augmentation complexity factor and target transformation factor
    augmented_memory_usage = image_memory_usage * aug_complexity_factor
    total_memory_usage = augmented_memory_usage + (image_memory_usage * target_complexity_factor)
    
    total_memory_per_worker = total_memory_usage + model_param_size
    return total_memory_per_worker


# Estimate augmentation complexity based on the types of transformations
def estimate_augmentation_complexity(transforms_compose):
    complexity_factor = 1.0  # Start with base complexity
    
    # Assign complexity factors based on common types of augmentations
    for t in transforms_compose.transforms:
        if isinstance(t, transforms.RandomResizedCrop):
            complexity_factor += 0.2
        elif isinstance(t, transforms.RandomHorizontalFlip):
            complexity_factor += 0.05
        elif isinstance(t, transforms.ColorJitter):
            complexity_factor += 0.3
        elif isinstance(t, transforms.RandomRotation):
            complexity_factor += 0.15
        # Add more augmentations as necessary
    
    return complexity_factor


# Estimate target transformation complexity
def estimate_target_transform_complexity(target_transforms):
    complexity_factor = 1.0  # Start with base complexity for labels
    
    # Assign complexity factors based on types of target transformations
    for t in target_transforms.transforms:
        if isinstance(t, transforms.Lambda):
            complexity_factor += 0.1  # Lambda transformations might be lightweight
        # Add more transformations and adjust complexity factors as necessary
    
    return complexity_factor


# Create model and get the number of parameters
def get_model_param_size(model_name):
    model = create_model(model_name, pretrained=True)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = model_params * 4  # 4 bytes per parameter (assuming float32)
    del model
    gc.collect()
    return model_size
    

# Function to calculate the memory usage per image (data and target transforms)
def estimate_memory_per_image(avg_image_size, aug_complexity_factor, target_complexity_factor):
    # Memory usage for image and target transformations
    image_memory_usage = avg_image_size * aug_complexity_factor
    target_memory_usage = avg_image_size * target_complexity_factor
    total_memory_usage = image_memory_usage + target_memory_usage
    return total_memory_usage


# Function to calculate the maximum batch size based on the number of workers and available RAM
def calculate_max_batch_size(image_folder, num_workers, model_name, data_transforms, target_transforms):
    # Get available system resources
    available_ram = get_available_ram()
    
    # Get model parameter size
    model_param_size = get_model_param_size(model_name)
    
    # Calculate average image size in the dataset folder and its subdirectories
    avg_image_size = calculate_average_image_size(image_folder)
    
    # Estimate augmentation complexity based on the provided data transforms
    aug_complexity_factor = estimate_augmentation_complexity(data_transforms)
    
    # Estimate target transform complexity
    target_complexity_factor = estimate_target_transform_complexity(target_transforms)
    
    # Estimate memory usage per image (with augmentation and target transform complexity factors)
    memory_per_image = estimate_memory_per_image(avg_image_size, aug_complexity_factor, target_complexity_factor)
    
    # Total memory used by the model itself
    total_model_memory = model_param_size
    
    # Subtract model memory from available RAM
    ram_available_for_data = available_ram - total_model_memory
    
    # Each worker needs memory to handle its portion of the batch
    memory_per_worker = memory_per_image * num_workers
    
    # Calculate the maximum number of images (batch size) that fits in the available memory
    max_batch_size = ram_available_for_data // memory_per_worker
    
    # Floor the batch size to the nearest multiple of 8
    floored_batch_size = (max_batch_size // 8) * 8
    
    return max(8, floored_batch_size)  # Ensure at least 8


# Main function to recommend the optimal number of workers
def recommend_num_workers(image_folder, batch_size, model_name, data_transforms, target_transforms):
    # Get available system resources
    available_ram = get_available_ram()
    num_cpu_cores = get_cpu_cores()
    
    # Get model parameter size
    model_param_size = get_model_param_size(model_name)
    
    # Calculate average image size in the dataset folder and its subdirectories
    avg_image_size = calculate_average_image_size(image_folder)
    
    # Estimate augmentation complexity based on the provided data transforms
    aug_complexity_factor = estimate_augmentation_complexity(data_transforms)
    
    # Estimate target transform complexity
    target_complexity_factor = estimate_target_transform_complexity(target_transforms)
    
    # Estimate memory usage per worker (with augmentation and target transform complexity factors)
    memory_per_worker = estimate_worker_memory_usage(
        avg_image_size, batch_size, model_param_size, aug_complexity_factor, target_complexity_factor
    )
    
    # Calculate the maximum number of workers that fit within available RAM
    max_workers_by_ram = available_ram // memory_per_worker
    
    # The number of workers shouldn't exceed the number of CPU cores
    recommended_workers = min(max_workers_by_ram, num_cpu_cores)
    
    return max(1, int(recommended_workers))  # Ensure at least 1 worker is recommended


if __name__ == "__main__":
    #num_workers = recommend_num_workers(image_folder, batch_size, model_name, data_transforms, target_transforms)
    #print(f"Recommended number of workers: {num_workers}")
    pass

