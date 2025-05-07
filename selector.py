import torch
from baseline_example import load_all_subject_data, compute_psd
import os
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1=nn.Linear(129 , 100)
        self.fc2=nn.Linear(100 , 84)
        self.fc3=nn.Linear(84 , 64)
        self.dt = nn.Dropout(0.5)
    
    def forward(self,input):
        input = input.float()
        out=F.relu(self.fc1(input))
        out=F.relu(self.fc2(out))
        out=self.fc3(out)
    
        return out

class Classifier_1(nn.Sequential):
    def __init__(self, emb_size=40, n_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            # nn.Linear(3904,64),
            nn.Linear(1920,64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(16, 4))
    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out

def standardize_data_1(X_train, X_test, channels):
    for j in range(channels):
        scaler = StandardScaler()
        X_train[:, j, :] = scaler.fit_transform(X_train[:, j, :])
        X_test[:, j, :] = scaler.transform(X_test[:, j, :])
    return X_train, X_test
def pretrain_encoder(eeg_data_M, labels_M, n_subjects, chans, num_epochs=10, batch_size=16):
    for subject_id in range(n_subjects):
        encoder = Encoder().to(device)
        classifier = Classifier_1().to(device)
        optimizer=torch.optim.Adam(list(encoder.parameters())+list(classifier.parameters()))
        loss_fn = nn.CrossEntropyLoss()
        org_data = np.nan_to_num(eeg_data_M[subject_id], nan=0.0, posinf=1e10, neginf=-1e10)
        org_labels = labels_M[subject_id]

        # 数据标准化
        org_data, _ = standardize_data_1(org_data, org_data, channels=chans)

        # 转换为Tensor
        org_data = torch.tensor(org_data, dtype=torch.float32)
        org_labels = torch.tensor(org_labels, dtype=torch.long)

        # 创建DataLoader
        org_dataset = TensorDataset(org_data, org_labels)
        org_loader = DataLoader(org_dataset, batch_size=batch_size, shuffle=True)

        # 训练Encoder
        for epoch in range(num_epochs):
            for data, labels in org_loader:
                data = data.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                encoded = encoder(data)
                y_pred = classifier(encoded)
                loss = loss_fn(y_pred, labels)
                loss.backward()
                optimizer.step()

        # 保存Encoder模型
        torch.save(encoder.state_dict(), f'{model_save_path}encoder_subject_{subject_id}.pth')
        print(f'Encoder for subject {subject_id} saved.')
        torch.save(classifier.state_dict(), f'{model_save_path}classifier_subject_{subject_id}.pth')
        print(f'Classifier for subject {subject_id} saved.')

# 调用预训练函数
if __name__ == '__main__':
    n_subjects = 29#15
    chans = 62  # 61/30
    n_classes = 2  # 2
    model_save_path = './saved_models/'  # './OLD_M_T_saved_models/' #'./Nback_saved_models/' #'./MA_saved_models/'
    base_path = 'input/input0/Per_Process_1/'
    eeg_data, labels = load_all_subject_data(
        n_subjects=n_subjects,
        compute_psd_fn=compute_psd,
        base_path=base_path,
        window_size=3000,
        step_size=500
    )
    pretrain_encoder(eeg_data, labels, n_subjects, chans)