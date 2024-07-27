import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class NNModel(nn.Module):
    # in_features, the size of the input vector for FFNN
    # h{n}: hidden layer 'n' size
    # out_features: the classifications count
    def __init__(self, in_features=13, h1 = 50, h2 = 100, h3=20, h4=5, out_features=1):
        super().__init__() # call nn.Module constructor
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h3, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.out(x))

        return x
    
    # predict given input, output prediction based on training
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).cpu()

# convert dataframe to train and test tensors.
# outputs X_train, X_test, Y_train, Y_test
def preprocess_dataframe(dataframe):
    # Split to input vector (X) and expected output (Y)

    dataframe.drop_duplicates()

    X = dataframe.drop('target', axis=1).values
    y = dataframe["target"].values
    
    # split train and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # convert to a Tensor
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    return X_train, X_test, y_train, y_test


def train_model(model : NNModel, dataframe, epochs=5000, gpu=False):

    x_train, x_test, y_train, y_test = preprocess_dataframe(dataframe)

    if gpu and torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Running on GPU: {torch.cuda.get_device_name(device)}")

        model = model.to(device)
        x_train = x_train.to(device)
        x_test = x_test.to(device)
        y_train = y_train.to(device)
        y_test = y_test.to(device)

    # Criterion to measure error
    criterion = nn.BCELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # start training
    for i in range(epochs):
        # forward the train to the model
        # squeeze the output to be of shape [y] instead of shape [y,1]
        y_pred = model.forward(x_train).squeeze(-1)

        loss = criterion(y_pred, y_train)

        if i % 100 == 0:
            print(f"Epoch: {i} and loss: {loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    evaluate_model(model, x_test, y_test)

    # once trained, return to CPU
    return model.cpu()


# Evaluate the model on test set
def evaluate_model(model, x_test, y_test):

    criterion = nn.BCELoss()
    correct = 0
    # evalulate training
    with torch.no_grad():
        y_eval = model.forward(x_test).squeeze(-1)
        loss = criterion(y_eval, y_test)
        print(f"Evalulation: {loss}")
        for i, data, in enumerate(x_test):
            y_val = model.forward(data)
            if (y_val < 0.5 and y_test[i] < 0.5) or (y_val >= 0.5 and y_test[i] > 0.5):
                correct += 1
        print(f"score: {correct}/{len(x_test)}")
    
    return correct/len(x_test)


def main():

    model = NNModel()

    df = pd.read_csv("dataset/heart.csv")

    model = train_model(model, df, gpu=True)

    print("Probability for heart disease: ", end="")
    print(model.predict(torch.tensor([25,1,0,140,211,1,0,165,1.4,1,1,1,3])))


if __name__ == "__main__":
    main()