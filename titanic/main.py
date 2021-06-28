"""
    get dataset from https://www.kaggle.com/c/titanic/overview
    submit_time: 2021/6/28
    submit_score: 0.76315
"""

import torch
import os
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split


#prepare dataset - 1
def get_dataset(file_path, features, drop_miss = True):
    if not isinstance(features, list) and not isinstance(features, tuple):
        raise Exception(f"features must be a list/tuple instead of {type(features)}")

    dataset = pd.read_csv(file_path, usecols=features)
    dataset["Sex"] = dataset["Sex"].map(lambda x: 1 if x == "male" else 0)

    for index, val in enumerate(dataset["Age"]):
        if not val > 0:
            if drop_miss:
                dataset.drop(index=index, inplace=True)
            else:
                dataset.at[index, "Age"] = 30

    return dataset.to_numpy(dtype=np.float32)


#prepare dataset - 2
class TitanicDataset(Dataset):
    def __init__(self, train_path):
        super(TitanicDataset, self).__init__()

        if not os.path.isfile(train_path):
            raise Exception("file not exist")

        self.dataset = torch.from_numpy(get_dataset(train_path, ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]))
        self.len = self.dataset.size(0)

    def __getitem__(self, item):
        return self.dataset[item][1:], self.dataset[item][[0]]

    def __len__(self):
        return self.len


#construct model
class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()

        #fc
        self.seq = nn.Sequential(
            nn.Linear(6, 4),
            nn.Sigmoid(),

            nn.Linear(4, 2),
            nn.Sigmoid(),

            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)


titanicDataset = TitanicDataset("./train.csv")
train_size = int(0.8 * len(titanicDataset))
test_size = len(titanicDataset) - train_size
batch_size = 64
#split dataset with target to train set and test set - for cross validation
train_dataset, test_dataset = random_split(titanicDataset, [train_size, test_size])

#loader - for mini-batch
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=2)

titanicModel = TitanicModel()

criterion = nn.BCELoss()
optimizer = optim.Adam(titanicModel.parameters(), lr=0.1)


# train func
def train(epoch, train_loader):
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #three steps for trainning
        #forward
        outputs = titanicModel(inputs)
        loss = criterion(outputs, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #update
        optimizer.step()

        running_loss += loss.item()

        if batch_idx:
            print("[%d, %5d] loss: %.2f" % (epoch+1, batch_idx+1, running_loss))
            running_loss = 0.0


# test func
def test(test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for(inputs, targets) in test_loader:
            outputs = (titanicModel(inputs) > 0.5).type(torch.float32)
            total += targets.size(0)
            correct += (outputs == targets).sum().item()

    print("test set accuracy: %.2f" % (100 * correct / total))


if __name__ == '__main__':
    ##train model
    for epoch in range(70):
        train(epoch, train_loader)
        test(test_loader)

    torch.save(titanicModel, './titanicModel.pth')  #save trained model

    ##use model to predict
    # titanicModel = torch.load("./titanicModel.pth")   #load trained model
    #
    # dataset = get_dataset("./test.csv", ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"], drop_miss=False)
    # passenger_id = dataset[:, 0]
    # features = dataset[:, 1:]
    #
    # predictions = titanicModel(torch.from_numpy(features))
    # with open("submission.csv", 'a') as f:
    #     f.write("PassengerId,Survived\n")
    #     for p_id, prediction in zip(passenger_id, predictions):
    #         f.write("%d,%d\n" % (int(p_id), 1 if prediction.item() >= 0.5 else 0))
