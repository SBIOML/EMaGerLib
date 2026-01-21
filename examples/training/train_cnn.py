import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pathlib import Path

from libemg.data_handler import OfflineDataHandler, RegexFilter
from libemg.feature_extractor import FeatureExtractor
from libemg.filtering import Filter

import emager_tools.models.models as etm
from emager_tools.config.loader import load_py_config

# Load configuration
cfg = load_py_config(Path(__file__).parent.parent / "config_examples" / "base_config_example.py")


def prepare_data(dataset_folder):
        classes_values = [str(num) for num in range(cfg.NUM_CLASSES)]
        reps_values = [str(num) for num in range(cfg.NUM_REPS)]
        regex_filters = [
            RegexFilter(left_bound="C_", right_bound="_", values=classes_values, description="classes"),
            RegexFilter(left_bound="R_", right_bound="_emg.csv", values=reps_values, description="reps"), 
            ]
        odh = OfflineDataHandler()
        odh.get_data(folder_location=dataset_folder, regex_filters=regex_filters)
        filter = Filter(cfg.SAMPLING)
        notch_filter_dictionary={ "name": "notch", "cutoff": 60, "bandwidth": 3}
        filter.install_filters(notch_filter_dictionary)
        bandpass_filter_dictionary={ "name":"bandpass", "cutoff": [20, 450], "order": 4}
        filter.install_filters(bandpass_filter_dictionary)
        filter.filter(odh)
        return odh

def main():
    data = prepare_data(cfg.DATASETS_PATH)
    # for i in range(len(data.data)):
    #     plt.plot(data.data[i])
    #     plt.show()

    # Split data into training and testing
    train_data = data.isolate_data("reps", [0,1,2])
    test_data = data.isolate_data("reps", [3,4])

    # Extract windows 
    train_windows, train_meta = train_data.parse_windows(cfg.WINDOW_SIZE, cfg.WINDOW_INCREMENT)
    test_windows, test_meta = test_data.parse_windows(cfg.WINDOW_SIZE, cfg.WINDOW_INCREMENT)

    print(f"Training metadata: {train_meta}, Testing metadata: {test_meta}")
    print(f"Training windows: {train_windows.shape}, Testing windows: {test_windows.shape}")


    # Features extraction
    # Extract MAV since it's a commonly used pipeline for EMG
    fe = FeatureExtractor()
    train_data = fe.getMAVfeat(train_windows)
    train_labels = train_meta["classes"]
    test_data = fe.getMAVfeat(test_windows)
    test_labels = test_meta["classes"]

    # pause for visualize features
    features_data = {"key": train_data}
    fe.visualize_feature_space(features_data, "PCA", classes=train_labels)

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(train_data.astype(np.float32)), torch.from_numpy(train_labels)),
        batch_size=64,
        shuffle=True,
    )
    test_dl = DataLoader(
        TensorDataset(torch.from_numpy(test_data.astype(np.float32)), torch.from_numpy(test_labels)),
        batch_size=256,
        shuffle=False,
    )

    # Fit and test the model
    classifier = etm.EmagerCNN((4, 16), cfg.NUM_CLASSES, -1)

    res = classifier.fit(train_dl, test_dl, max_epochs=cfg.EPOCH)
    acc = int(res[0]["test_acc"]*1000)
    print(f"Resultat: {res} accuracy : {acc}/1000")
    current_time = datetime.datetime.now().strftime("%y-%m-%d_%Hh%M")
    print(f"Current time: {current_time}")

    # Save the model
    model_path = f"{cfg.SAVE_PATH}libemg_torch_cnn_{cfg.SESSION}_{acc}_{current_time}.pth"
    torch.save(classifier.state_dict(), model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    main()
print(f"Model saved at {model_path}")