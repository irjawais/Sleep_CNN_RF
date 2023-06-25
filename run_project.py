#https://chat.openai.com/share/ae2d38ce-2730-4a36-b76f-ec34de637a89
import os
import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Initialize empty lists to store the features and labels
features = []
labels = []

# Loop over each subject's directory
for subject_dir in os.listdir('./training data'):
    # Skip any files in the 'training data' directory
    if not os.path.isdir(os.path.join('./training data', subject_dir)):
        continue

    # Loop over each set of EEG data in the subject's directory
    for file in os.listdir(os.path.join('./training data', subject_dir, 'data')):
        # Skip any files that are not .vhdr files
        if not file.endswith('.vhdr'):
            continue

        # Load the EEG data
        raw = mne.io.read_raw_brainvision(os.path.join('./training data', subject_dir, 'data', file), preload=True)

        # Load the events from the marker file
        events, event_id = mne.events_from_annotations(raw)

        # Print the event_id dictionary
        print(event_id)

        # Loop over the events
        for i in range(len(events) - 1):
            # If the current event is an error
            if events[i, 2] == event_id['Stimulus/S 96']:
                # If the next event is a response
                if events[i + 1, 2] == event_id['Stimulus/S 80']:
                    # Label as 'correct'
                    labels.append(1)
                else:
                    # Label as 'incorrect'
                    labels.append(0)

        # Extract epochs around the error event (S96)
        epochs = mne.Epochs(raw, events, event_id=event_id['Stimulus/S 96'], tmin=-0.1, tmax=1.0, baseline=(None, 0))
        print(epochs.get_data().shape)
        # Compute the Power Spectral Density (PSD) for each epoch
        psd, freqs = mne.time_frequency.psd_array_multitaper(epochs.get_data(), sfreq=epochs.info['sfreq'])
        
        # Average across all channels
        psd_avg = np.mean(psd, axis=1)
        
        # Reshape the averaged PSD values to a 2D feature matrix and add it to the list of features
        features.append(psd.reshape(len(epochs), -1))

# Convert the lists of features and labels to arrays
features = np.concatenate(features)

labels = np.array(labels)
print(features.shape, labels.shape)

data = np.column_stack((features, labels))
np.savetxt('dataset.csv', data, delimiter=',')


X = features
y = labels

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

''' oversample = SMOTE()
X, y = oversample.fit_resample(X, y) '''

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize the SVM classifier
clf = svm.SVC()

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred)
print(report)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, cohen_kappa_score

# Create a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Predict labels for the training and test sets
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Calculate evaluation metrics for the training set
accuracy_train = accuracy_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)
auc_train = roc_auc_score(y_train, y_train_pred)
kappa_train = cohen_kappa_score(y_train, y_train_pred)

# Calculate evaluation metrics for the test set
accuracy_test = accuracy_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(y_test_pred, y_test)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
ax.set_title('RF Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()


# Print the evaluation metrics
print("Training Set Metrics:")
print(f"Accuracy: {accuracy_train:.4f}")
print(f"Recall: {recall_train:.4f}")
print(f"Precision: {precision_train:.4f}")
print(f"F1-Score: {f1_train:.4f}")
print(f"AUC: {auc_train:.4f}")
print(f"Cohen's Kappa: {kappa_train:.4f}")

print("\nTest Set Metrics:")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"F1-Score: {f1_test:.4f}")


import numpy as np
from imblearn.over_sampling import SMOTE
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import networkx as nx
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns
from scipy.spatial.distance import euclidean
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score, roc_auc_score, auc
from torchviz import make_dot
import netron
import scikitplot as skplt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import plotly.graph_objects as go


def create_adj_matrix(graph):
    num_nodes = len(graph)
    adj_matrix = [[0 for i in range(num_nodes)] for j in range(num_nodes)]

    for node in graph:
        for neighbor in graph[node]:
            adj_matrix[node][neighbor] = 1
    return adj_matrix

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        x = torch.relu(x)
        x = self.conv5(x, edge_index)
        x = torch.relu(x)
        return x

class LSTM(nn.Module):
    def __init__(self, in_features, hidden_size, num_classes):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        x = self.fc(h_n[-1, :, :])
        return x

class GCN_LSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden_size, num_classes):
        super(GCN_LSTM, self).__init__()
        self.gcn = GCN(in_channels, hidden_channels, out_channels)
        self.lstm = LSTM(out_channels, hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        x = torch.unsqueeze(x, 0)
        x = self.lstm(x)
        return x
    

def create_graph(X, y):
    
    # Create an empty graph
    G = nx.Graph()
    # Add each row as a node in the graph
    for i in range(X.shape[0]):
        G.add_node(i)
    # Loop through all nodes to create edges
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if i == j:
                continue
            dist = euclidean(X[i], X[j])

            if dist < 0.2 : #
                G.add_edge(i, j)
    return  G

G = create_graph(X,y)
adj = nx.adjacency_matrix(G)
adj = adj.todense()
adj = np.array(adj)


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(adj, labels, test_size=0.2, random_state=42)
print("===>",G.number_of_nodes())
# Create the GCN + LSTM model
model = GCN_LSTM(in_channels=G.number_of_nodes(), hidden_channels=120, out_channels=16, hidden_size=120, num_classes=2)
print(model)
from torchsummary import summary
summary(model)
num_nodes = X_train.shape[0]
valid_indices = np.where((X_train.nonzero()[0] < num_nodes) & (X_train.nonzero()[1] < num_nodes))
edge_index = torch.from_numpy(np.concatenate((X_train.nonzero()[0][valid_indices][np.newaxis, :],
                                                X_train.nonzero()[1][valid_indices][np.newaxis, :]), axis=0))
                                                
adj = torch.from_numpy(adj)
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)

print(X_train.shape,G.number_of_nodes(),edge_index.shape)
accuracies = []
recalls = []
precisions = []
losses = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

y_train = torch.from_numpy(y_train).long()

y_train_pred = []
for epoch in range(1500):
    optimizer.zero_grad()
    out = model(X_train.float(), edge_index)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    _, predicted = torch.max(out.data, 1)
    y_train_pred.append(predicted)
    correct = (predicted == y_train).sum().item()
    accuracy = correct / len(y_train)
    accuracies.append(accuracy)
    recall = recall_score(y_train, predicted)
    recalls.append(recall)
    precision = precision_score(y_train, predicted)
    f1 = f1_score(y_train, predicted)
    sensitivity = recall



    k = cohen_kappa_score(y_train, predicted)
    auc_value = roc_auc_score(y_train, predicted)
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-measure: {f1:.4f}, Sensitivity: {sensitivity:.4f},  k-cohen: {k:.4f}, AUC: {auc_value:.4f}')

out = out.detach().numpy()
y_train = y_train.detach().numpy()

tsne = TSNE(n_components=2)
embeddings_tsne = tsne.fit_transform(out)
plt.scatter(embeddings_tsne[:,0], embeddings_tsne[:,1], c=y_train)
plt.show()



# Assuming you have separate X_test and y_test datasets for testing

y_test = torch.from_numpy(y_test).long()  # Convert y_test to PyTorch tensor

y_test_pred = []

# Evaluate the model on the test data
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    out = model(X_test.float(), edge_index)
    test_loss = criterion(out, y_test)
    _, predicted_test = torch.max(out.data, 1)
    y_test_pred.append(predicted_test)
    correct_test = (predicted_test == y_test).sum().item()
    accuracy_test = correct_test / len(y_test)
    recall_test = recall_score(y_test, predicted_test)
    precision_test = precision_score(y_test, predicted_test)
    f1_test = f1_score(y_test, predicted_test)
    sensitivity_test = recall_test
    k_test = cohen_kappa_score(y_test, predicted_test)
    print(f'Test Results - Loss: {test_loss.item():.4f}, Accuracy: {accuracy_test:.4f}, Recall: {recall_test:.4f}, Precision: {precision_test:.4f}, F1-measure: {f1_test:.4f}, Sensitivity: {sensitivity_test:.4f}, k-cohen: {k_test:.4f}')
print(y_test_pred,y_test)

''' from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(y_test_pred, y_test)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
ax.set_title('GNN LSTM Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show() '''


test_features = []
# Loop over each subject's directory
for subject_dir in os.listdir('./test data'):
    # Skip any files in the 'training data' directory
    if not os.path.isdir(os.path.join('./test data', subject_dir)):
        continue

    # Loop over each set of EEG data in the subject's directory
    for file in os.listdir(os.path.join('./test data', subject_dir, 'data')):
        # Skip any files that are not .vhdr files
        if not file.endswith('.vhdr'):
            continue

        # Load the EEG data
        raw = mne.io.read_raw_brainvision(os.path.join('./test data', subject_dir, 'data', file), preload=True)

        # Load the events from the marker file
        events, event_id = mne.events_from_annotations(raw)


        # Extract epochs around the error event (S96)
        epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=1.0, baseline=(None, 0))
        print(epochs.get_data().shape)
        # Compute the Power Spectral Density (PSD) for each epoch
        psd, freqs = mne.time_frequency.psd_array_multitaper(epochs.get_data(), sfreq=epochs.info['sfreq'])
        
        # Average across all channels
        psd_avg = np.mean(psd, axis=1)
        
        # Reshape the averaged PSD values to a 2D feature matrix and add it to the list of features
        test_features.append(psd.reshape(len(epochs), -1))

# Convert the lists of features and labels to arrays
test_features = np.concatenate(test_features)


y_train_pred = rf_model.predict(test_features)
print(y_train_pred)