import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from trajectory_dataset_multistep import TrajectoryImageDatasetMultistep
from tqdm import tqdm
from loss_functions import * 
from raster_scenario import save_rgb_tensor_image
import segmentation_models_pytorch as smp

# -------- Train/Validation Function -------- #
def run_epoch(model, dataloader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(dataloader, desc="Training" if is_train else "Validating"):
            inputs = batch[0]  # shape: [B, 4, H, W]
            targets = batch[1:]  # list of tensors: [B, 4, H, W] for each step

            inputs = inputs.to(device)
            targets = [t.to(device) for t in targets]

            loss = 0
            preds = []

            x = inputs
            for t in targets:
                pred = model(x)
                preds.append(pred)
                loss += criterion(pred, t)
                x = pred.detach()  # Feed predicted as input

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(dataloader)

def denormalize(img: torch.Tensor) -> torch.Tensor:
    return ((img.clamp(-1, 1) + 1) / 2 * 255).round().clamp(0, 255).to(torch.uint8)

def save_predictions(model, dataloader, output_dir):
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = batch[0].to(device)                      # (B, 3, H, W)
            targets = [t.to(device) for t in batch[1:]]       # list of (B, 3, H, W)
            B = inputs.size(0)

            for b in range(B):  # Loop over batch elements
                x_b = inputs[b].unsqueeze(0)  # (1, 3, H, W)
                scenario_dir = output_dir / f"scenario_{i * B + b}"
                scenario_dir.mkdir(parents=True, exist_ok=True)

                for step_idx, target in enumerate(targets):
                    pred = model(x_b)         # (1, 3, H, W)
                    pred_b = pred[0]          # (3, H, W)
                    target_b = target[b]      # (3, H, W)
                    
                    # Save images
                    save_rgb_tensor_image(denormalize(pred_b), scenario_dir / f"pred_step_{step_idx}.png")
                    save_rgb_tensor_image(denormalize(target_b), scenario_dir / f"gt_step_{step_idx}.png")

                    # Autoregressive input update
                    x_b = pred

if __name__ == "__main__":
    # --- Settings ---
    epochs = 200
    batch_size = 4
    num_steps = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ])

    # --- Datasets ---
    train_set = TrajectoryImageDatasetMultistep("converted_dataset", "train", num_steps, transform=transform)
    val_set = TrajectoryImageDatasetMultistep("converted_dataset", "val", num_steps, transform=transform)
    test_set = TrajectoryImageDatasetMultistep("converted_dataset", "test", num_steps, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # --- Model, Loss, Optimizer ---
    model = smp.Unet(
        encoder_name='resnet34', 
        encoder_weights='imagenet', 
        in_channels=3,                 
        classes=3
    ).to(device)
    criterion = balanced_weighted_l1_loss
    model_name = "Unet"
    loss_name = "balanced_weighted_l1"
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    results_dir = Path("results") / f"{model_name}_{loss_name}"
    model_path = results_dir / "best_model.pt"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model name:", model_name)
    print(f"Loss: ", loss_name)

    # --- Training Loop ---
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion)
        test_output_dir = results_dir / str(epoch)
        save_predictions(model, val_loader, test_output_dir)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model at epoch {epoch}")

    # --- Test and Save Predictions ---
    print("\nTesting best model...")
    model.load_state_dict(torch.load(model_path))
    test_output_dir = results_dir / str(epochs)
    save_predictions(model, test_loader, test_output_dir)
    print(f"Saved predictions to {test_output_dir}")
