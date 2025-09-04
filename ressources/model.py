import torch
from torch import nn
from math import sqrt


class MlpModel(nn.Module):
    def __init__(self, input_dim: int):
        super(MlpModel, self).__init__()
        self.proj = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.regression = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.proj(x)
        x = self.relu(x)
        x = self.regression(x)
        return x


def train_loop(device, dataloader, ml_model, loss_fn, optimizer):
    ml_model.train()  # set model to train mode

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # Compute prediction error
        pred = ml_model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()



def test_loop(device, validation_loader, training_loader, ml_model, loss_fn, n_epoch):
    num_batches_train = len(training_loader)
    num_batches_val = len(validation_loader)

    ml_model.eval()

    test_loss = 0
    train_loss = 0

    with torch.no_grad():
        for X_val, y_val in validation_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            pred = ml_model(X_val)
            test_loss += loss_fn(pred, y_val).item()

        for X_train, y_train in training_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            pred = ml_model(X_train)
            train_loss += loss_fn(pred, y_train).item()

    test_loss /= num_batches_val
    train_loss /= num_batches_train
    print(f"Epoch {n_epoch+1} RMSE: {sqrt(train_loss):>8f} (Training), {sqrt(test_loss):>8f} (Validation)")
    return sqrt(train_loss), sqrt(test_loss)

