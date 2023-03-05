import torch
from torchvision import transforms
from livelossplot import PlotLosses
from functions import *


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    liveplot = PlotLosses()
    for i, (img, params) in enumerate(train_loader):
        img, params = img.to(device), params.to(device)
        img = img.float().to(device)
        params = params.float().to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, params)
        loss.backward()
        optimizer.step()
    return loss / len(train_loader.dataset)


def test(model, test_loader, criterion, device, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (img, params) in enumerate(test_loader):
            img, params = img.to(device), params.to(device)
            img = img.float().to(device)
            params = params.float().to(device)
            output = model(img)
            loss = criterion(output, params)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f"Epoch {epoch}, Average test loss: {avg_test_loss}")    
    return avg_test_loss

