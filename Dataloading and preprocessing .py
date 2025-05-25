import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from imblearn.over_sampling import SMOTE
from PyEMD import CEEMDAN
from sklearn.model_selection import train_test_split

# Load sample ECG data (replace with your dataset)
def load_ecg_data():
    # Example: MIT-BIH dataset (simulated for brevity)
    X = np.random.randn(1000, 256)  # 1000 samples, 256 timesteps
    y = np.random.randint(0, 5, 1000)  # 5 classes
    return X, y

# CEEMDAN Denoising
def ceemdan_denoise(signal):
    ceemd = CEEMDAN()
    imfs = ceemd(signal)
    denoised = np.sum(imfs[:-1], axis=0)  # Exclude residual
    return denoised

# Preprocessing pipeline
X_raw, y = load_ecg_data()
X_denoised = np.array([ceemdan_denoise(x) for x in X_raw])

# Apply SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_denoised, y)
