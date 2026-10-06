import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
from Backbone import GaussianNoise, MagnitudeScaling, TemporalShift, ChannelDropout, ChannelRotation

data = np.loadtxt('data/SGT/me/day1/C_0_R_1_emg.csv', delimiter=',')

window = data[800:1000]


fig, axs = plt.subplots(3, 2)

# Original signal
axs[0, 0].plot(window)
axs[0, 0].set_title('Original Signal')

# Gaussian Noise
gaussian_noise = GaussianNoise(snr_db=10)
noisy_window = gaussian_noise(window.copy().T)
axs[0, 1].plot(noisy_window.T)
axs[0, 1].set_title('Gaussian Noise')

# Magnitude Scaling
magnitude_scaling = MagnitudeScaling((0.5, 2))
scaled_window = magnitude_scaling(window.copy().T)
axs[1, 0].plot(scaled_window.T)
axs[1, 0].set_title('Magnitude Scaling')

# Temporal Shift
temporal_shift = TemporalShift()
shifted_window = temporal_shift(window.copy().T)
axs[1, 1].plot(shifted_window.T)
axs[1, 1].set_title('Temporal Shift')

# Channel Dropout
channel_dropout = ChannelDropout(dropout_prob=0.1)
dropped_window = channel_dropout(window.copy().T)
axs[2, 0].plot(dropped_window.T)
axs[2, 0].set_title('Channel Dropout')

# Channel Rotation
channel_rotation = ChannelRotation()
rotated_window = channel_rotation(window.copy().T)
axs[2, 1].plot(rotated_window.T)
axs[2, 1].set_title('Channel Rotation')

plt.tight_layout()
plt.show()

# Plot training losses for both models
def load_losses_from_checkpoints(checkpoint_dir):
    """Load train and val losses from checkpoint files."""
    checkpoint_files = sorted(glob.glob(f'{checkpoint_dir}/checkpoint_epoch_*.pt'))
    epochs = []
    train_losses = []
    val_losses = []

    for ckpt_path in checkpoint_files:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        epochs.append(ckpt['epoch'])
        train_losses.append(ckpt['train_loss'])
        if ckpt['val_loss'] is not None:
            val_losses.append(ckpt['val_loss'])

    return epochs, train_losses, val_losses if val_losses else None

# Load losses for both models
adv_epochs, adv_train, adv_val = load_losses_from_checkpoints('trained_models/adversarial/checkpoints')
no_adv_epochs, no_adv_train, no_adv_val = load_losses_from_checkpoints('trained_models/no_adversarial/checkpoints')

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot training losses
axs[0].plot(no_adv_epochs, no_adv_train, 'b-o', label='No Adversarial')
axs[0].plot(adv_epochs, adv_train, 'r-o', label='Adversarial')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('SimCLR Loss')
axs[0].set_title('Training Loss')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Plot validation losses
if no_adv_val and adv_val:
    axs[1].plot(no_adv_epochs, no_adv_val, 'b-o', label='No Adversarial')
    axs[1].plot(adv_epochs, adv_val, 'r-o', label='Adversarial')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('SimCLR Loss')
    axs[1].set_title('Validation Loss')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
