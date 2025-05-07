
#example:swin_transformer(in_subjects & finetune)
import os,random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mne
from scipy.signal import welch
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
from timm.models.swin_transformer import SwinTransformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def extract_time_windows(eeg_data, labels, window_size, step_size):
    n_channels, n_timepoints = eeg_data.shape
    time_windows, window_labels = [], []
    for start in range(0, n_timepoints - window_size + 1, step_size):
        end = start + window_size
        time_windows.append(eeg_data[:, start:end])
        window_labels.append(labels)
    return np.array(time_windows), np.array(window_labels)

def compute_psd(x):
    _, psd = welch(x, nperseg=256, axis=1)
    return psd

def standardize_data(X_train, X_test, channels):
    for j in range(channels):
        scaler = StandardScaler()
        X_train[:, 0, j, :] = scaler.fit_transform(X_train[:, 0, j, :])
        X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])
    return X_train, X_test

def load_all_subject_data(n_subjects, compute_psd_fn, base_path, window_size=3000, step_size=500):
    eeg_data_all, labels_all = [], []

    for sub in range(n_subjects):
        subject_id = f"sub-0{sub+1}" if sub < 9 else f"sub-{sub+1}"
        subject_path = os.path.join(base_path, subject_id, "ses-S1", "eeg")

        data_all, label_all = [], []
        for task, label_value in zip(["easy", "med", "diff"], [0, 1, 2]):
            file_path = os.path.join(subject_path, f"MATB{task}_eeg.fif")
            raw = mne.io.read_raw_fif(file_path, preload=True)
            eeg_data, _ = raw[:]
            label_array = label_value * np.ones(eeg_data.shape[0])
            data, labels = extract_time_windows(eeg_data, label_array, window_size, step_size)
            data = compute_psd_fn(data.transpose(0, 2, 1)).transpose(0, 2, 1)
            data_all.append(data)
            label_all.append(labels)

        subject_data = np.concatenate(data_all)
        subject_labels = np.concatenate(label_all)[:, 0]
        eeg_data_all.append(subject_data)
        labels_all.append(subject_labels)

    return np.array(eeg_data_all), np.array(labels_all)
def build_loader(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long))
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    )

def build_model(n_classes=2, chans=30, samples=129):
    return SwinTransformer(
        img_size=(chans, samples),
        patch_size=(3, 3),
        in_chans=1,
        num_classes=n_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
    )

def train_model(model, train_loader, test_loader, epochs=10, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.unsqueeze(1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return model

def swin_transformer_classifier(model, X_train, y_train, X_val, y_val, X_test, y_test,
                                n_classes=2, chans=30, samples=129,
                                batch_size=64, epochs=10, device='cuda'):
    train_loader, val_loader, test_loader = build_loader(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)
    trained_model = train_model(model, train_loader, val_loader, test_loader, epochs, device)
    return trained_model

def train_within_subject(eeg_data, labels, build_model, swin_transformer_classifier, device,
                         n_subjects=29, chans=62, n_classes=3):
    random_number = random.randint(1, 100)
    for sub in range(n_subjects):
        features, label = eeg_data[sub], labels[sub]
        kf = KFold(n_splits=50, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(features):
            X_train, X_test = features[test_idx], features[train_idx]
            y_train, y_test = label[test_idx], label[train_idx]
            X_train, X_test = standardize_data(X_train, X_test, channels=chans)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.111, random_state=42)

            X_train, y_train = shuffle(X_train, y_train, random_state=random_number)
            X_val, y_val = shuffle(X_val, y_val, random_state=random_number)
            X_test, y_test = shuffle(X_test, y_test, random_state=random_number)

            model = build_model(n_classes, chans, samples=eeg_data[0].shape[-1]).to(device)
            swin_transformer_classifier(model, X_train, y_train, X_val, y_val, X_test, y_test)


def train_cross_subject_with_finetune(eeg_data, labels, build_model, swin_transformer_classifier, device,
                                      n_subjects=29, chans=62, n_classes=3, finetune_ratio=0.01):
    random_number = random.randint(1, 100)
    kf = KFold(n_splits=n_subjects)
    for train_idx, test_idx in kf.split(np.arange(n_subjects)):
        print('Target subject:', test_idx)

        X_train = np.vstack(eeg_data[train_idx])
        y_train = np.hstack(labels[train_idx])
        X_test = np.vstack(eeg_data[test_idx])
        y_test = labels[test_idx][0]

        X_train, X_test = standardize_data(X_train, X_test, channels=chans)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.111, random_state=42)

        X_train, y_train = shuffle(X_train, y_train, random_state=random_number)
        X_val, y_val = shuffle(X_val, y_val, random_state=random_number)
        X_test, y_test = shuffle(X_test, y_test, random_state=random_number)

        model = build_model(n_classes, chans, samples=eeg_data[0].shape[-1]).to(device)
        model = swin_transformer_classifier(model, X_train, y_train, X_val, y_val, X_test, y_test)

        # fine-tune
        X_finetune_train, X_test_final, y_finetune_train, y_test_final = train_test_split(
            X_test, y_test, test_size=1 - finetune_ratio, random_state=42)

        if len(X_finetune_train) < 2:
            print(f"[Skip fine-tuning] Subject {test_idx} has too few samples for fine-tuning.")
            continue

        X_finetune_train, X_finetune_val, y_finetune_train, y_finetune_val = train_test_split(
            X_finetune_train, y_finetune_train, test_size=0.1, random_state=42)

        X_finetune_train, y_finetune_train = shuffle(X_finetune_train, y_finetune_train, random_state=random_number)
        X_finetune_val, y_finetune_val = shuffle(X_finetune_val, y_finetune_val, random_number)
        X_test_final, y_test_final = shuffle(X_test_final, y_test_final, random_number)

        swin_transformer_classifier(model, X_finetune_train, y_finetune_train,
                                    X_finetune_val, y_finetune_val, X_test_final, y_test_final)

if __name__ == '__main__':
    n_subjects = 29
    base_path = 'input/input0/Per_Process_1/'
    eeg_data, labels = load_all_subject_data(
        n_subjects=n_subjects,
        compute_psd_fn=compute_psd,
        base_path=base_path,
        window_size=3000,
        step_size=500
    )
    train_within_subject(eeg_data, labels, build_model, swin_transformer_classifier, device)
    train_cross_subject_with_finetune(eeg_data, labels, build_model, swin_transformer_classifier, device)
