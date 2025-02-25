import torch
import pandas
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn as nn
import torch.nn.functional as F
import optuna
import torch.optim as optim
from optuna.trial import TrialState
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
dataset = pandas.read_csv("dataset/tea.csv",header = None)
output_dir = "output/tea"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def mean_normalization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std
def apply_mean_normalization(data, mean, std):
    return (data - mean) / std
def expand_to_channels(data, num_channels=8):
    data_expanded = data[:, np.newaxis, :]  # 扩展维度到 (num_samples, 1, num_features)
    data_expanded = np.repeat(data_expanded, num_channels, axis=1)
    return data_expanded
data = dataset.iloc[0: ,2:].values
label = dataset.iloc[0: ,1].values
data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.3, random_state=42)
label_encoder = LabelEncoder()
label_train_y = label_encoder.fit_transform(label_train)
label_test_y = label_encoder.fit_transform(label_test)
data_train_normalized, mean, std = mean_normalization(data_train)
data_test_normalized, mean, std = mean_normalization(data_test)
data_train_expanded = expand_to_channels(data_train_normalized, num_channels=1)
data_test_expanded = expand_to_channels(data_test_normalized, num_channels=1)
data_train_tensor = torch.tensor(data_train_expanded, dtype=torch.float32).to(device)
data_test_tensor = torch.tensor(data_test_expanded, dtype=torch.float32).to(device)
label_train_tensor = torch.tensor(label_train_y, dtype=torch.long).to(device)
label_test_tensor = torch.tensor(label_test_y, dtype=torch.long).to(device)
train_dataset = TensorDataset(data_train_tensor, label_train_tensor)
test_dataset = TensorDataset(data_test_tensor, label_test_tensor)
print(data_train_tensor.shape)
print(data_test_tensor.shape)

class CNN1D(nn.Module):
    def __init__(self, num_classes, filters, dense1_units, dense2_units, activation=nn.ReLU):
        super(CNN1D, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filters[0], kernel_size=9)
        self.bn1 = nn.BatchNorm1d(filters[0])
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=filters[0], out_channels=filters[1], kernel_size=7)
        self.bn2 = nn.BatchNorm1d(filters[1])
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=filters[1], out_channels=filters[2], kernel_size=5)
        self.bn3 = nn.BatchNorm1d(filters[2])
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv1d(in_channels=filters[2], out_channels=filters[3], kernel_size=5)
        self.bn4 = nn.BatchNorm1d(filters[3])
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv1d(in_channels=filters[3], out_channels=filters[4], kernel_size=5)
        self.bn5 = nn.BatchNorm1d(filters[4])
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv1d(in_channels=filters[4], out_channels=filters[5], kernel_size=3)
        self.bn6 = nn.BatchNorm1d(filters[5])
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv1d(in_channels=filters[5], out_channels=filters[6], kernel_size=3)
        self.bn7 = nn.BatchNorm1d(filters[6])
        self.pool7 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv1d(in_channels=filters[6], out_channels=filters[7], kernel_size=1)
        self.bn8 = nn.BatchNorm1d(filters[7])
        self.pool8 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv1d(in_channels=filters[7], out_channels=filters[8], kernel_size=1)
        self.bn9 = nn.BatchNorm1d(filters[8])
        self.pool9 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv10 = nn.Conv1d(in_channels=filters[8], out_channels=filters[9], kernel_size=1)
        self.bn10 = nn.BatchNorm1d(filters[9])
        self.pool10 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(filters[-1], dense1_units)
        self.fc2 = nn.Linear(dense1_units, dense2_units)
        self.fc3 = nn.Linear(dense2_units, num_classes)

    def forward(self, x):
        x = self.pool1(self.activation()(self.bn1(self.conv1(x))))
        x = self.pool2(self.activation()(self.bn2(self.conv2(x))))
        x = self.pool3(self.activation()(self.bn3(self.conv3(x))))
        x = self.pool4(self.activation()(self.bn4(self.conv4(x))))
        x = self.pool5(self.activation()(self.bn5(self.conv5(x))))
        x = self.pool6(self.activation()(self.bn6(self.conv6(x))))
        x = self.pool7(self.activation()(self.bn7(self.conv7(x))))
        x = self.pool8(self.activation()(self.bn8(self.conv8(x))))
        x = self.pool9(self.activation()(self.bn9(self.conv9(x))))
        x = self.pool10(self.activation()(self.bn10(self.conv10(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.activation()(self.fc1(x)))
        x = self.activation()(self.fc2(x))
        x = self.activation()(self.fc3(x))
        return x

def objective(trial):
    activation_function_str = trial.suggest_categorical('activation_function', ['ELU', 'ReLU', 'Sigmoid'])
    activation_function_map = {
        'ELU': nn.ELU,
        'ReLU': nn.ReLU,
        'Sigmoid': nn.Sigmoid
    }
    activation_fn = activation_function_map[activation_function_str]
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'RMSprop'])
    filters = [
        trial.suggest_int('filters_1', 4, 64),
        trial.suggest_int('filters_2', 4, 64),
        trial.suggest_int('filters_3', 8, 128),
        trial.suggest_int('filters_4', 16, 256),
        trial.suggest_int('filters_5', 16, 256),
        trial.suggest_int('filters_6', 16, 256),
        trial.suggest_int('filters_7', 16, 256),
        trial.suggest_int('filters_8', 8, 128),
        trial.suggest_int('filters_9', 8, 128),
        trial.suggest_int('filters_10', 4, 64)
    ]
    dense1_units = trial.suggest_int('dense1_units', 64, 512)
    dense2_units = trial.suggest_int('dense2_units', 64, 512)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 4, 64)
    # num_epochs = trial.suggest_int('epochs', 10, 100, step=10)
    num_epochs = 60
    num_folds = 5

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_train_losses = []
    fold_test_accuracies = []
    fold_test_stds = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(train_dataset)):
        train_subset = Subset(train_dataset, train_idx)
        test_subset = Subset(train_dataset, test_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        num_classes = len(np.unique(train_idx))
        model = CNN1D(num_classes, filters=filters, activation=activation_fn, dense1_units=dense1_units,
                      dense2_units=dense2_units).to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        if optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr)

        epoch_test_accuracies = []
        epoch_test_std = []
        epoch_train_losses = []

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_train_losses.append(running_loss / len(train_loader))

            model.eval()
            correct_test = 0
            total_test = 0
            all_accuracies = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs_test = model(inputs)
                    _, predicted_test = torch.max(outputs_test.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted_test == labels).sum().item()
            accuracy = correct_test / total_test * 100
            all_accuracies.append(accuracy)
            epoch_accuracy = np.mean(all_accuracies)
            epoch_std = np.std(all_accuracies)
            epoch_test_accuracies.append(epoch_accuracy)
            epoch_test_std.append(epoch_std)

        fold_accuracies.append(np.mean(epoch_test_accuracies))
        fold_train_losses.append(epoch_train_losses)
        fold_test_accuracies.append(epoch_test_accuracies)
        fold_test_stds.append(epoch_test_std)

    trial.set_user_attr('fold_train_losses', fold_train_losses)
    trial.set_user_attr('fold_test_accuracies', fold_test_accuracies)
    trial.set_user_attr('fold_test_stds', fold_test_stds)

    average_accuracy = np.mean(fold_accuracies)
    trial.set_user_attr('average_accuracy', average_accuracy)

    return average_accuracy
num_trials = 40
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=num_trials)
trials = study.trials
cumulative_best_accuracies = []
for i in range(1, len(trials) + 1):
    accuracies = [t.value for t in trials[:i]]
    cumulative_best_accuracies.append(max(accuracies))
params = []
best_trial = study.best_trial
for key, value in best_trial.params.items():
    params.append(value)
activation_function_map = {
        'ELU': nn.ELU,
        'ReLU': nn.ReLU,
        'Sigmoid': nn.Sigmoid
    }
activation_function = activation_function_map[ params[0]]

def plot_tsne(features, labels, fold, save_path):
    features = np.array(features)
    colors = [
        '#1f77b4', 
        '#ff7f0e',  
        '#2ca02c',  
        '#d62728',  
        '#9467bd',  
        '#8c564b',  
        '#e377c2',  
        '#7f7f7f',  
        '#bcbd22',  
        '#17becf',  
        '#f7b733',  
        '#f45d42', 
        '#e0e0e0',  
        '#ffb74d',  
        '#64b5f6',  
        '#81c784',  
        '#ba68c8',  
        '#ff8a65',  
        '#90caf9',  
        '#a5d6a7',  
        '#ce93d8',  
        '#ffcc80',  
        '#d7ccc8',  
        '#cfd8dc',  
        '#ffb300',  
        '#8d6e63'  
    ]
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    tsne_result = tsne.fit_transform(features)
    plt.figure(figsize=(12, 8))
    for label in np.unique(labels):
        idxs = labels == label
        plt.scatter(tsne_result[labels == label, 0], tsne_result[labels == label, 1],
                    label=str(label), alpha=0.7, color=colors[label % len(colors)], edgecolors='k')
        center_x = np.mean(tsne_result[idxs, 0])
        center_y = np.mean(tsne_result[idxs, 1])
        plt.text(center_x, center_y, str(label), fontsize=12, fontweight='bold',
                 color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    class_names = [str(i) for i in range(1, 25)]
    plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('t-SNE 2D Component1', fontsize=14, fontweight='bold')
    plt.ylabel('t-SNE 2D Component2', fontsize=14, fontweight='bold')
    plt.title(f't-SNE Visualization', fontsize=16, fontweight='bold')
    plt.savefig(f'{save_path}/tsne_fold_{fold}.png')
    plt.close()
def plot_tsne_3D(features, labels, fold, save_path):
    features = np.array(features)  # 将列表转换为 NumPy 数组
    # labels = np.array(labels)  # 将列表转换为 NumPy 数组
    colors = [
        '#1f77b4',  
        '#ff7f0e',  
        '#2ca02c',  
        '#d62728', 
        '#9467bd',  
        '#8c564b',  
        '#e377c2',  
        '#7f7f7f',  
        '#bcbd22', 
        '#17becf',
        '#f7b733',  
        '#f45d42',  
        '#e0e0e0',  
        '#ffb74d',  
        '#64b5f6',  
        '#81c784',  
        '#ba68c8', 
        '#ff8a65',  
        '#90caf9',  
        '#a5d6a7',  
        '#ce93d8',  
        '#ffcc80', 
        '#d7ccc8',  
        '#cfd8dc',  
        '#ffb300',
        '#8d6e63'  
    ]
    tsne = TSNE(n_components=3, random_state=42, perplexity=50)
    tsne_result = tsne.fit_transform(features)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter_objects = []
    for label in np.unique(labels):
        scatter = ax.scatter(tsne_result[labels == label, 0], tsne_result[labels == label, 1],
                             tsne_result[labels == label, 2],
                             label=str(label), alpha=0.7, color=colors[label % len(colors)], edgecolors='k')
        scatter_objects.append(scatter)
    class_names = [str(i) for i in range(1, 25)]
    legend1 = ax.legend(handles=scatter.legend_elements()[0], labels=class_names, title="Classes")
    ax.add_artist(legend1)
    ax.set_xlabel('t-SNE Component 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Component 2', fontsize=14, fontweight='bold')
    ax.set_zlabel('t-SNE Component 3', fontsize=14, fontweight='bold')
    ax.set_title('3D t-SNE Visualization', fontsize=16, fontweight='bold')
    # ax.set_title(f'5um')
    plt.savefig(f'{save_path}/tsne_fold_{fold}_3.png')
    plt.close()
def train_model(model, criterion, optimizer, train_loader, test_loader, epochs=5, save_path=output_dir):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    tsne_features = []
    tsne_labels = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
        train_losses.append(train_loss / len(train_loader))
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        predictions = []
        true_labels = []
        features = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()
                predictions.extend(predicted.tolist())
                true_labels.extend(targets.tolist())
                features.append(outputs.cpu().numpy())
        test_losses.append(test_loss / len(test_loader))
        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)
        cm = confusion_matrix(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.6f}, "
              f"Train Accuracy: {train_accuracy:.6f}%, Test Loss: {test_losses[-1]:.6f}, "
              f"Test Accuracy: {test_accuracy:.6f}%, Precision: {precision:.6f}, "
              f"Recall: {recall:.6f}, F1 Score: {f1:.6f}")
        tsne_features.extend(np.concatenate(features))
        tsne_labels.extend(true_labels)
    plot_tsne(tsne_features, tsne_labels, fold, save_path)
    plot_tsne_3D(tsne_features, tsne_labels, fold, save_path)
    torch.save(model.state_dict(), os.path.join(output_dir, 'tea_model_k.pth'))
    return train_losses, test_losses, train_accuracies, test_accuracies, cm

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
all_train_losses = []
all_test_losses = []
all_train_accuracies = []
all_test_accuracies = []
all_confusion_matrices = []
for train_index, test_index in kf.split(train_dataset):
    fold += 1
    print(f"Fold {fold}")
    if max(train_index) >= len(train_dataset) or max(test_index) >= len(train_dataset):
        print(
            f"索引超出范围: train_index={max(train_index)}, test_index={max(test_index)}, 数据集大小={len(train_dataset)}")
        continue
    train_subset = Subset(train_dataset, train_index)
    test_subset = Subset(train_dataset, test_index)
    train_loader = DataLoader(train_subset, batch_size=params[-1], shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=params[-1], shuffle=False)
    num_classes = 24
    model = CNN1D(num_classes, filters=params[2:12], dense1_units=params[-4], dense2_units=params[-3],
                  activation=activation_function).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    lr = params[-2]
    optimizer_name = params[1]
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    train_losses, test_losses, train_accuracies, test_accuracies, cm = train_model(
        model, criterion, optimizer, train_loader, test_loader, epochs=60)
    all_train_losses.append(train_losses)
    all_test_losses.append(test_losses)
    all_train_accuracies.append(train_accuracies)
    all_test_accuracies.append(test_accuracies)
    all_confusion_matrices.append(cm)

mean_train_losses = np.mean(all_train_losses, axis=0)
mean_test_losses = np.mean(all_test_losses, axis=0)
mean_train_accuracies = np.mean(all_train_accuracies, axis=0)
mean_test_accuracies = np.mean(all_test_accuracies, axis=0)

print(f"Mean Train Losses: {mean_train_losses}")
print(f"Mean Test Losses: {mean_test_losses}")
print(f"Mean Train Accuracies: {mean_train_accuracies}")
print(f"Mean Test Accuracies: {mean_test_accuracies}")


train_loader = DataLoader(train_dataset, batch_size=params[-1], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=params[-1], shuffle=False)
num_classes = len(np.unique(label_train))
model = CNN1D(num_classes,filters=params[2:12],dense1_units=params[-5],dense2_units=params[-4],activation=activation_function).to(device)
criterion = nn.CrossEntropyLoss().to(device)
lr = params[-2]
optimizer_name = params[1]
if optimizer_name == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif optimizer_name == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

train_losses, test_losses, train_accuracies, test_accuracies,cm = train_model(
    model, criterion, optimizer, train_loader, test_loader, epochs=60)
epochs = range(1, len(train_losses) + 1)
