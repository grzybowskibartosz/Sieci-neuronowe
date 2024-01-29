import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

df = pd.read_csv('D:\Sieci neuronowe\Diabetes problem\diabetes.csv')

df_sum = df.isnull().sum()

num_of_preg = df['Pregnancies']

X = df.drop("Outcome", axis=1).values ### indepenedent features
y = df['Outcome'].values ### dependent features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

### Creating tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

### Creating model with Pytorch

class ANN_Model(nn.Module):
    def __init__(self, input_feature=8, hidden1=20, hidden2=20, out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_feature,hidden1)
        self.f_connected2 = nn.Linear(hidden1,hidden2)
        self.out          = nn.Linear(hidden2,out_features)

    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))

        x = self.out(x)
        return x

### Instantiate my ANN_model
    
torch.manual_seed(20)
model = ANN_Model()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs =50000
final_losses=[]
for i in range(epochs):
    i += 1
    y_pred = model.forward(X_train)
    loss = loss_function(y_pred, y_train)
    final_losses.append(loss.item())
    if i%100 == 1:
        print("Epoch number: {} and the loss: {}".format(i, loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# plt.plot(range(epochs), final_losses)
# plt.ylabel("Loss")
# plt.xlabel("Epoch")

# plt.show()

# predictions = []
# with torch.no_grad():
#     for i, data in enumerate(X_test):
#         y_pred=model(data)
#         predictions.append(y_pred.argmax().item())
#         print(y_pred.argmax().item())


# cm = confusion_matrix(y_test, predictions)

# plt.figure(figsize=(10, 6))
# sns.heatmap(cm, annot=True)
# plt.xlabel('Actual values')
# plt.ylabel('Predicted values')
# plt.show()

new_data = [1,115,70,30,96,34.6,0.529,32]
new_data2 = [7,196,90,0,0,39.8,0.451,41]
new_data3 = [10,125,70,26,115,31.1,0.205,41]
new_data4 = [3,158,76,36,245,31.6,0.851,28]
new_data5 = [13,145,82,19,110,22.2,0.245,57]


new_tensor = torch.tensor(new_data)
new_tensor2 = torch.tensor(new_data2)
new_tensor3 = torch.tensor(new_data3)
new_tensor4 = torch.tensor(new_data4)
new_tensor5 = torch.tensor(new_data5)



with torch.no_grad():
    print(model(new_tensor))
    print(model(new_tensor).argmax().item())
    print(model(new_tensor2))
    print(model(new_tensor2).argmax().item())
    print(model(new_tensor3))
    print(model(new_tensor3).argmax().item())
    print(model(new_tensor4))
    print(model(new_tensor4).argmax().item())
    print(model(new_tensor5))
    print(model(new_tensor5).argmax().item())