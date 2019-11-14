import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import Adagrad, Adam, SGD
from torch.utils import data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from input_handler import final_embedding_data_loader



vec_file_1 = str(sys.argv[0])
vec_file_2 = str(sys.argv[1])
cognate_file = str(sys.argv[2])
class_report = bool(sys.argv[3])
acc_report = bool(sys.argv[4])


################################# HyperParameters #################################
#batch = 64
#input_dim = 100
#hidden_dim = 82
#output_dim = 64

input_data, final_prepared_data, input_X, labels = final_embedding_data_loader(vec_file_1, vec_file_2, cognate_file)



X_train, X_test, y_train, y_test = train_test_split(input_X, labels, test_size = 0.15)
X_train, X_val, y_train, y_val = train_test_split(input_X, labels, test_size = 0.1)

def feed_to_tensor(tensor, data):
    for i, el in enumerate(data):
        tensor[i] = el


final_tensor_train = torch.empty([len(X_train), 2, 100])
feed_to_tensor(final_tensor_train, X_train)

labels = torch.empty(len(y_train))
feed_to_tensor(labels, y_train)

final_tensor_test = torch.empty([len(X_val), 2, 100])
feed_to_tensor(final_tensor_test, X_val)

labels_val = torch.empty(len(y_val))
feed_to_tensor(labels_val, y_val)

final_model_val = torch.empty(len(X_test), 2, 100)
feed_to_tensor(final_model_val, X_test)

final_labels = torch.empty(len(y_test))
feed_to_tensor(final_labels, y_test)


train = torch.utils.data.TensorDataset(final_tensor_train, labels)
train_loader = torch.utils.data.DataLoader(train, 
                                           batch_size = 64,
                                           shuffle = True)
test = torch.utils.data.TensorDataset(final_tensor_test, labels_val)
test_loader = torch.utils.data.DataLoader(test, 
                                          batch_size = 64,
                                          shuffle = True)

train_final = torch.utils.data.TensorDataset(final_model_val, final_labels)
final_test_loader = torch.utils.data.DataLoader(train_final, 
                                                   batch_size = 64,
                                                   shuffle = True)  


class SiameseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SiameseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, trace = None):
#         out = []
        
        out1 = self.fc1(x[:, 0, :])
        out1 = self.relu(out1)
        out1 = self.fc2(out1)

        out2 = self.fc1(x[:, 1, :])
        out2 = self.relu(out2)
        out2 = self.fc2(out2)
        
        if trace:
            print(out1.size(), out2.size())
            
        
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        logits = cos(out1, out2)

        #manhatten = torch.dist(out1, out2, p=2)
        
        #sigm_logits = torch.sigmoid(logits)
        #preds = torch.round(sigm_logits)
        
#         preds = self.prediction(out)
        
        return logits


input_dim = 100
hidden_dim = 82
output_dim = 64

device = torch.device('cuda:3')

model = SiameseFeedForward(input_dim, hidden_dim, output_dim)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss() 
optimizer = Adam(model.parameters())

for epoch in range(50):  #num_epochs
    losses = []
    for X, label in train_loader:
        
        X = X.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        #print(X.shape)
        #print(label.shape)

        scores = model(X, trace=False)  
        #print(scores.shape)
        #print(label.shape)

        #sigm_logits = torch.sigmoid(scores)
        #preds = torch.round(sigm_logits)

        loss = criterion(preds, label)
        #losses.append(loss.item())
        
        loss.backward()
        
        optimizer.step()
        

        with torch.no_grad():

            true_labels = []
            pred_labels = []

            for X_val, labels_val in test_loader:

                X_val = X_val.to(device)
                labels_val = labels_val.to(device)

                scores = model(X_val)

                sigm_logits = torch.sigmoid(scores)
                preds = torch.round(sigm_logits)

                true_labels += labels_val.cpu().detach().numpy().tolist()
                pred_labels += preds.cpu().detach().numpy().tolist()
    
        
    print(np.mean(losses))
    if class_report:
        print(classification_report(true_labels, pred_labels))
    if acc_report:
        print(accuracy_score(true_labels, pred_labels))





true_lab = []
pred_lab = []

for X, label in final_test_loader:
    X = X.to(device)
    label = label.to(device)
    
    scores = model(X)
    
    sigm_logits = torch.sigmoid(scores)
    preds = torch.round(sigm_logits)
    
    true_lab += label.cpu().detach().numpy().tolist()
    pred_lab += preds.cpu().detach().numpy().tolist()


print(accuracy_score(true_lab, pred_lab))
print(confusion_matrix(true_lab, pred_lab))
print(classification_report(true_lab, pred_lab))

