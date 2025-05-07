import os
import numpy as np
import math
import random
import itertools
from baseline_example import load_all_subject_data, compute_psd, standardize_data
from selector import standardize_data_1, Encoder, Classifier_1
import time
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import torch
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Feature_Extractor_Enc(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16))
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(20, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(0.5))
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), groups=32, bias=False),
            nn.Conv2d(32, 32, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(0.5))

        self.projection = nn.Sequential(
            nn.Conv2d(32, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'), )

    def forward(self, X: Tensor) -> Tensor:
        x, y = X[0], X[1]
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.projection(x)

        y = self.firstconv(y)
        y = self.depthwiseConv(y)
        y = self.separableConv(y)
        y = self.projection(y)
        return (x, y)


class MultiHeadAttention_Enc_Dec(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, X, mask: Tensor = None) -> Tensor:
        x_enc, x_dec = X[1], X[0]  # enc is target, dec is source
        queries = rearrange(self.queries(x_dec), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x_dec), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x_enc), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd_Dec1(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, X, **kwargs):
        x, y = X[0], X[1]
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return (x, y)


class ResidualAdd_Dec2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.lm = nn.LayerNorm(40)

    def forward(self, X, **kwargs):
        x, y = X[0], X[1]
        res = x
        x = self.lm(x)
        x = self.fn((x, y), **kwargs)
        x += res
        return (x, y)


def visualize_feature_label_relationship(data, labels):
    flattened_data = data.reshape(data.shape[0], -1)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(flattened_data)
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette="Set2")
    plt.title("PCA of EEG Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Label")
    plt.show()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1)))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.cuda.FloatTensor(np.ones(d_interpolates.shape)), requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True, )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class Transformer(nn.Sequential):
    def __init__(self, depth=1, emb_size=40, num_heads=2, drop_p=0.5, forward_expansion=2, forward_drop_p=0.5):
        decoder_block = nn.Sequential(
            ResidualAdd_Dec2(nn.Sequential(
                MultiHeadAttention_Enc_Dec(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p))),
            ResidualAdd_Dec1(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p))))
        super().__init__(*[decoder_block for _ in range(depth)])


class Feature_Extractor(nn.Sequential):
    def __init__(self, emb_size=40, depth=1):
        super().__init__(
            Feature_Extractor_Enc(emb_size),
            Transformer(depth, emb_size))


class Classifier(nn.Sequential):
    def __init__(self, emb_size=40, n_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(6880, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class Discriminator(nn.Sequential):
    def __init__(self, emb_size=40, n_classes=2, **kwargs):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, 20),
            nn.ELU(),
            nn.Linear(20, n_classes))

    def forward(self, x):
        x = self.clshead(x)
        return x


class DCD(nn.Module):
    def __init__(self, h_features=64, input_features=128):
        super(DCD, self).__init__()

    def forward(self, inputs):
        out = F.relu(self.fc1(inputs))
        out = self.fc2(out)
        return F.softmax(self.fc3(out), dim=1)


class DATrans():
    def __init__(self, nsub):
        super(DATrans, self).__init__()
        self.batch_size = 64
        self.n_epochs = 10
        self.n_epochs_1 = 1
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (172, 40)
        self.lambda_cen = 0.5
        self.lambda_cls = 2
        self.lambda_gp = 10
        self.alpha = 0.0002
        self.nSub = nsub
        self.root = './'
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.Feature_Extractor = nn.DataParallel(Feature_Extractor()).cuda()
        self.Classifier = nn.DataParallel(Classifier()).cuda()
        self.Discriminator = nn.DataParallel(Discriminator()).cuda()
        self.centers = {}

    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(2):
            cls_idx = np.where(label == cls4aug)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 8), 1, 61, 129))
            for ri in range(int(self.batch_size / 8)):
                for rj in range(3):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 3)
                    tmp_aug_data[ri, :, :, rj * 43:(rj + 1) * 43] = tmp_data[rand_idx[rj], :, :, rj * 43:(rj + 1) * 43]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 8)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def update_centers(self, feature, label):
        deltac = {}
        count = {}
        count[0] = 0
        for i in range(len(label)):
            l = label[i]
            if l in deltac:
                deltac[l] += self.centers[l] - feature[i]
            else:
                deltac[l] = self.centers[l] - feature[i]
            if l in count:
                count[l] += 1
            else:
                count[l] = 1

        for ke in deltac.keys():
            deltac[ke] = deltac[ke] / (count[ke] + 1)

        return deltac

    def train(self, X_train, y_train, X_val, y_val, X_test, y_test):

        self.Feature_Extractor.apply(weights_init_normal)
        self.Classifier.apply(weights_init_normal)
        self.Discriminator.apply(weights_init_normal)
        # 获取数据
        sour_img = X_train
        sour_label = y_train

        # fine-tune
        X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(X_test, y_test, test_size=0.99,
                                                                                random_state=42)
        X_test_train, y_test_train = shuffle(X_test_train, y_test_train, random_state=random_number)
        X_test_test, y_test_test = shuffle(X_test_test, y_test_test, random_state=random_number)

        n_samples = X_test_train.shape[0]
        new_n_samples = sour_img.shape[0]
        weights = np.random.rand(n_samples)  # 随机为每个样本分配权重
        weights = weights / np.sum(weights)  # 归一化权重
        img = np.zeros((new_n_samples, 1, 61, 129))  # 1,62,129;1,
        label = np.zeros((new_n_samples,))
        for i in range(new_n_samples):
            random_indices = np.random.choice(n_samples, size=2, replace=False)  # 随机选择2个样本
            sample_1 = X_test_train[random_indices[0], :, :, :]
            sample_2 = X_test_train[random_indices[1], :, :, :]
            label_1 = y_test_train[random_indices[0]]
            label_2 = y_test_train[random_indices[1]]
            img[i, :, :, :] = weights[0] * sample_1 + weights[1] * sample_2  # 根据权重生成新样本
            if label_1 == label_2:
                label[i] = label_1
            else:
                label[i] = label_1 if weights[0] > weights[1] else label_2

        val_data = X_val
        val_label = y_val
        test_data = X_test_test
        test_label = y_test_test

        sour_img = torch.from_numpy(sour_img)
        sour_label = torch.from_numpy(sour_label)
        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        val_data = torch.from_numpy(val_data)
        val_label = torch.from_numpy(val_label)

        dataset = torch.utils.data.TensorDataset(img, label, sour_img, sour_label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True,
                                                      shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size,
                                                           drop_last=True, shuffle=True)

        for i in range(self.c_dim):
            self.centers[i] = torch.randn(self.dimension).cuda()
            self.centers[i] = self.centers[i]

        # Optimizers
        self.optimizer_F = torch.optim.Adam(self.Feature_Extractor.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.Feature_Extractor.parameters(), self.Classifier.parameters()), lr=self.lr,
            betas=(self.b1, self.b2))
        self.optimizer_dis = torch.optim.Adam(self.Discriminator.parameters(), lr=0.0004, betas=(self.b1, self.b2))
        averAcc = 0
        num = 0
        gamma = 1
        best_acc_val = 0

        # --------------
        #  Train the Feature_Extractor  & domain discriminator
        for e in range(5):
            tacc = 0
            tnum = 0
            self.Feature_Extractor.train()
            self.Discriminator.train()

            for i, (img, label, sour_img, sour_label) in enumerate(self.dataloader):
                img = Variable(img.type(self.Tensor))
                label = Variable(label.type(self.LongTensor))
                sour_img = Variable(sour_img.type(self.Tensor))
                sour_label = Variable(sour_label.type(self.LongTensor))

                if (i + 1) % 1 == 0:
                    self.optimizer_dis.zero_grad()
                    with torch.no_grad():
                        (feature, sour_feature) = self.Feature_Extractor((img, sour_img))
                    # discriminator
                    pre_dom = self.Discriminator(feature.detach())
                    pre_dom_sour = self.Discriminator(sour_feature.detach())
                    gradient_penalty = compute_gradient_penalty(self.Discriminator, sour_feature, feature)
                    loss_D = -torch.mean(pre_dom) + torch.mean(pre_dom_sour) + self.lambda_gp * gradient_penalty
                    self.optimizer_dis.step()

                if (i + 1) % 2 == 0:
                    self.optimizer_F.zero_grad()
                    (feature, sour_feature) = self.Feature_Extractor((img, sour_img))
                    domain_pred_for_adv = self.Discriminator(sour_feature)
                    loss_F_adv = - torch.mean(domain_pred_for_adv)  # 对源域错误的分类越多越好
                    loss_F_adv.backward()
                    self.optimizer_F.step()

        # --------------
        #  Train Feature_Extractor and the classifier+loss_Dis
        # --------------
        for e in range(30):
            tacc = 0
            tnum = 0
            self.Feature_Extractor.eval()
            self.Classifier.train()
            self.Discriminator.eval()
            for i, (img, label, sour_img, sour_label) in enumerate(self.dataloader):
                img = Variable(img.type(self.Tensor))
                label = Variable(label.type(self.LongTensor))
                sour_img = Variable(sour_img.type(self.Tensor))
                sour_label = Variable(sour_label.type(self.LongTensor))

                if (i + 1) % 1 == 0:

                    self.optimizer.zero_grad()
                    aug_data, aug_label = self.interaug(X_test, y_test)
                    img = torch.cat((img[:48], aug_data))
                    label = torch.cat((label[:48], aug_label))
                    (sour_feature, feature) = self.Feature_Extractor((sour_img, img))
                    # classifier
                    out_cls = self.Classifier(feature)
                    sour_out_cls = self.Classifier(sour_feature)
                    # discriminator
                    pre_cls_fake = self.Discriminator(sour_feature)  # 对源域错误的分类

                    # Classification loss
                    loss_cls_targ = self.criterion_cls(out_cls, label)
                    loss_cls_sour = self.criterion_cls(sour_out_cls, sour_label)
                    x1 = loss_cls_targ
                    x2 = loss_cls_sour
                    lambda_1 = 0.1
                    lambda_2 = 1.0
                    epsilon = 0.5
                    gap = torch.abs(x1 - x2)
                    loss_Joint_cls = x1 + x2 + lambda_1 * (x1 - x2) ** 2 + lambda_2 * torch.max(torch.tensor(0.0),
                                                                                                gap - epsilon)

                    # loss_Joint_cls = loss_cls_targ + loss_cls_sour
                    loss_Joint_adv = - torch.mean(pre_cls_fake)
                    loss_U = loss_Joint_cls + loss_Joint_adv
                    loss_U.backward()
                    self.optimizer.step()

                    # Training accuracy for target data
                    for tk in range(len(label)):
                        tnum = tnum + 1
                        train_pred = torch.max(out_cls, 1)[1]
                        if train_pred[tk] == label[tk]:
                            tacc = tacc + 1

            if (e + 1) % 1 == 0:
                self.Feature_Extractor.eval()
                self.Classifier.eval()

                with torch.no_grad():
                    vacc = 0
                    vnum = 0
                    val_data = Variable(val_data.type(self.Tensor))
                    val_label = Variable(val_label.type(self.LongTensor))
                    (_, vfeature) = self.Feature_Extractor((val_data, val_data))
                    vCls = self.Classifier(vfeature)
                    y_pred = torch.max(vCls, 1)[1]

                    for k in range(len(val_label)):
                        vnum = vnum + 1
                        if y_pred[k] == val_label[k]:
                            vacc = vacc + 1
                    vacc = 1.0 * vacc / vnum

                    if vacc > best_acc_val:
                        best_acc_val = vacc
                        best_epoch = e

        with torch.no_grad():
            n_classes = 2
            # 用于存储混淆矩阵的统计量（三分类）
            confusion_matrix = torch.zeros(n_classes, n_classes, dtype=torch.long)  # 假设是3分类任务
            acc = 0
            num = 0

            test_data = Variable(test_data.type(self.Tensor))
            test_label = Variable(test_label.type(self.LongTensor))
            (feature, feature) = self.Feature_Extractor((test_data, test_data))
            # visualize_feature_label_relationship(feature.cpu(),test_label.cpu())

            train_data = Variable(torch.from_numpy(X_train[:3000]).type(self.Tensor))
            train_label = Variable(torch.from_numpy(y_train[:3000]).type(self.LongTensor))
            (feature1, feature1) = self.Feature_Extractor((train_data, train_data))
            # visualize_feature_label_relationship(feature1.cpu(),train_label.cpu())

            Cls = self.Classifier(feature)
            y_pred = torch.max(Cls, 1)[1]
            for k in range(len(test_label)):
                num = num + 1
                if y_pred[k] == test_label[k]:
                    acc = acc + 1
            acc = 1.0 * acc / num

            acc1 = 0
            num1 = 0
            Cls1 = self.Classifier(feature1)
            y_pred1 = torch.max(Cls1, 1)[1]
            for k in range(len(train_label)):
                num1 = num1 + 1
                if y_pred1[k] == train_label[k]:
                    acc1 = acc1 + 1
            acc1 = 1.0 * acc1 / num1

            # 填充混淆矩阵
            for t, p in zip(test_label, y_pred):
                confusion_matrix[t, p] += 1

            print("Confusion Matrix:\n", confusion_matrix)

            SEN = []
            SPE = []
            F1 = []
            for cls in range(n_classes):
                TP = confusion_matrix[cls, cls]
                FN = torch.sum(confusion_matrix[cls, :]) - TP
                FP = torch.sum(confusion_matrix[:, cls]) - TP
                TN = torch.sum(confusion_matrix) - TP - FN - FP
                print(f"Class {cls}: TP={TP}, FN={FN}, FP={FP}, TN={TN}")
                sen = TP.float() / (TP + FN) if (TP + FN) != 0 else 0.0  # 灵敏度/召回率
                SEN.append(sen)
                spe = TN.float() / (TN + FP) if (TN + FP) != 0 else 0.0  # （特异度）
                SPE.append(spe)
                precision = TP.float() / (TP + FP) if (TP + FP) != 0 else 0.0  # Precision
                f1 = 2 * (precision * sen) / (precision + sen) if (precision + sen) != 0 else 0.0
                F1.append(f1)  # F1 Score

            averAcc = averAcc / num
            print('The test accuracy is:', acc)
            print('The train accuracy is:', acc1)
            macro_SEN = sum(SEN) / len(SEN)
            macro_SPE = sum(SPE) / len(SPE)
            macro_F1 = sum(F1) / len(F1)

            print('Per-class SEN:', SEN)
            print('Per-class SPE:', SPE)
            print('Per-class F1:', F1)
            print('Macro SEN:', macro_SEN)
            print('Macro SPE:', macro_SPE)
            print('Macro F1:', macro_F1)
            # return best_epoch, acc, macro_SEN, macro_SPE, macro_F1
        return best_epoch, acc


def load_encoder_classifier(subject_id):
    model_save_path = './OLD_M_T_saved_models/'  # './saved_models/' #'./Nback_saved_models/' #'./MA_saved_models/'
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load(f'{model_save_path}encoder_subject_{subject_id}.pth'))
    classifier = Classifier_1().to(device)
    classifier.load_state_dict(torch.load(f'{model_save_path}classifier_subject_{subject_id}.pth'))
    return encoder, classifier


if __name__ == '__main__':
    best = 0
    aver = 0
    n_subjects = 15  # 29
    chans = 61  # 62,
    kf = KFold(n_splits=n_subjects)  # 留一个被试作为测试集
    random.seed(time.time())
    random_number = random.randint(1, 100)
    n_classes = 2  # 3
    model_save_path = './saved_models/'  # './OLD_M_T_saved_models/' #'./Nback_saved_models/' #'./MA_saved_models/'
    base_path = 'input/input0/Per_Process_1/'
    eeg_data, labels = load_all_subject_data(
        n_subjects=n_subjects,
        compute_psd_fn=compute_psd,
        base_path=base_path,
        window_size=3000,
        step_size=500
    )
    for train_index, test_index in kf.split(np.arange(n_subjects)):
        seed_n = np.random.randint(2024)
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        print('target:', test_index)
        # -————————————————————————————————————————————————————————————————————————————————
        n = 14  # 选择准确率最高的 n 个模型
        accuracies = []
        best_models_indices = []

        for i in train_index:
            X_test = np.nan_to_num(eeg_data[test_index][0], nan=0.0, posinf=1e10, neginf=-1e10)
            y_test = labels[test_index][0]
            X_test, _ = standardize_data_1(X_test, X_test, channels=chans)
            tar_data, _, tar_labels, _ = train_test_split(X_test, y_test, test_size=0.985, random_state=42)
            tar_data = torch.tensor(tar_data, dtype=torch.float32)
            tar_labels = torch.tensor(tar_labels, dtype=torch.long)
            tar_dataset = TensorDataset(tar_data, tar_labels)
            tar_loader = DataLoader(tar_dataset, batch_size=16, shuffle=True)
            encoder, classifier = load_encoder_classifier(i)
            acc = 0
            for data, labels in tar_loader:
                data = data.to(device)
                labels = labels.to(device)
                y_test_pred = classifier(encoder(data))
                acc = acc + (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()
            accuracy = round(acc / len(tar_loader), 3)
            accuracies.append(accuracy)
            best_models_indices.append(i)
        top_n_indices = np.argsort(accuracies)[-n:]  # 获取准确率最高的 n 个索引
        print(accuracies)
        best_models_indices = [np.array([idx]) if np.isscalar(idx) else idx for idx in best_models_indices]
        train_index = np.concatenate([idx.flatten() for idx in best_models_indices])
        train_index = np.concatenate([best_models_indices[i] for i in top_n_indices])
        print(train_index)

        X_train = np.vstack(eeg_data[train_index])[np.newaxis, :].transpose(1, 0, 2, 3)
        y_train = np.hstack(labels[train_index])
        X_test = np.vstack(eeg_data[test_index])[np.newaxis, :].transpose(1, 0, 2, 3)
        y_test = labels[test_index][0]
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)

        X_train, X_test = standardize_data(X_train, X_test, channels=chans)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.111, random_state=42)

        X_train, y_train = shuffle(X_train, y_train, random_state=random_number)
        X_val, y_val = shuffle(X_val, y_val, random_state=random_number)
        X_test, y_test = shuffle(X_test, y_test, random_state=random_number)

        datrans = DATrans(test_index)  # 选择目标域
        best_epoch, acc = datrans.train(X_train, y_train, X_val, y_val, X_test, y_test)
        aver = aver + acc

aver = aver / n_subjects
print('The average Aver accuracy is: ' + str(aver) + "\n")