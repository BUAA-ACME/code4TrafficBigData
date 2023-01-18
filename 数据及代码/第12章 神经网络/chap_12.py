import numpy as np  
  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.optim import Adam  
  
from torch.utils.data import Dataset, DataLoader, Subset  
  
  
class TrafficDataset(Dataset):  
    def __init__(self, path, x_offset=12, y_offset=6):  
        self._path = path  
        df = torch.from_numpy(np.loadtxt(path, delimiter=',', dtype=np.float32))  
        tod = torch.arange(df.shape[0])  
        tod %= 12*24  
        tod = tod.float() / (12*24)  
        tod = tod.reshape(-1, 1)  
        tod = tod.expand(df.shape)  
        df = torch.stack([df, tod], dim=-1)  
        xs = []  
        ys = []  
        for i in range(len(df)-x_offset-y_offset):  
            x = df[i:i+x_offset]  
            y = df[i+x_offset:i+x_offset+y_offset, :, :1]  
            xs.append(x)  
            ys.append(y)  
        self.x_data = torch.stack(xs)  
        self.y_data = torch.stack(ys)  
  
    def __len__(self):  
        return self.x_data.shape[0]  
  
    def __getitem__(self, item):  
        return self.x_data[item], self.y_data[item]  
  
  
class DenseNet(nn.Module):  
    def __init__(self):  
        super(DenseNet, self).__init__()  
        self.dense = nn.Sequential(  
            nn.Linear(21*12*2, 64),  
            nn.ReLU(),  
            nn.Linear(64, 64),  
            nn.ReLU(),  
            nn.Linear(64, 21*6)  
        )  
  
    def forward(self, x):  
        shape = x.shape  
        x = x.reshape(shape[0], -1)  
        return self.dense(x).reshape(shape[0], 6, shape[2], 1)  
  
data = TrafficDataset("traffic_speed_chap12&13.csv")  
train_len = int(len(data) * 0.6)  
val_len = int(len(data) * 0.8)  
train_data = Subset(data, range(train_len))  
val_data = Subset(data, range(train_len, val_len))  
test_data = Subset(data, range(val_len, len(data)))  
  
# datashape(batchsize, seq_len: 12, sensor_len: 21, size:2)  
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)  
val_loader = DataLoader(val_data, batch_size=32)  
test_loader = DataLoader(test_data, batch_size=32)  
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  
net = DenseNet().to(device)  
cretic = nn.L1Loss()  
optimizer = Adam(net.parameters(), lr=3e-4, weight_decay=1e-5)  
best = 99999  
for i in range(1,150+1):  
    train_loss = []  
    val_loss = []  
    for x,y in train_loader:  
        net.train()  
        x = x.to(device)  
        y = y.to(device)  
        optimizer.zero_grad()  
        loss = cretic(y, net(x))  
        loss.backward()  
        optimizer.step()  
        train_loss.append(loss.item())  
  
    for x, y in val_loader:  
        net.eval()  
        x = x.to(device)  
        y = y.to(device)  
        loss = cretic(y, net(x))  
        val_loss.append(loss.item()) 
    print(i, "train_loss:{}, val_loss:{}".format(np.mean(train_loss), np.mean(val_loss)))  
    if np.mean(val_loss)<best:  
        best = np.mean(val_loss)  
        best_state_dict = net.state_dict()  
  
print("best_val_loss:", best)  
net.load_state_dict(best_state_dict)  
test_loss = []  
for x, y in test_loader:  
    net.eval()  
    x = x.to(device)  
    y = y.to(device)  
    loss = cretic(y, net(x))  
    test_loss.append(loss.item())  
  
print("test:{}".format(np.mean(test_loss)))  
torch.save(net.state_dict(), 'parameter_best_Dense_{:.6f}.pkl'.format(np.mean(test_loss)))  