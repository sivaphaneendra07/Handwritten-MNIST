from sklearn.datasets import load_digits
import torch
import matplotlib.pyplot as plt
digits = load_digits()

# first image
image = digits.images[9]

# label
label = digits.target[100]

print("Label:", label)

plt.imshow(image, cmap="gray")
plt.show()

x = digits.data/16
#x = torch.tensor(x,dtype=torch.float32)
y = digits.target
#y = torch.tensor(y,dtype=torch.long)
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test = tts(
    x,
    y,
    test_size = 0.2,
    random_state = 42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

import torch.nn as nn

model = nn.Sequential(
    nn.Linear(64,32),nn.ReLU(),
    nn.Linear(32,16),nn.ReLU(),
    nn.Linear(16,10)
) 

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)

for epoch in range(10000):
    y_pred = model(x_train)
    loss = loss_fn(y_pred,y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch%1000 == 0:
        print(loss.item())
y_test_pred = model(x_test)
    
from sklearn.metrics import accuracy_score as acc
predicted = torch.argmax(
        y_test_pred,
        dim=1
)
accu = acc(
    y_test.numpy(),
    predicted.numpy()
)
print(accu)    

