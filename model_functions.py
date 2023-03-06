import torch
from torchvision import transforms
from livelossplot import PlotLosses
from functions import *


def train(model, train_loader, optimizer, criterion, device):
    '''
    Train function for our model
    '''
    model.train()  # set to training mode
    liveplot = PlotLosses()  # To visualise the loss plots
    for i, (img, params) in enumerate(train_loader):
        img, params = img.to(device), params.to(device)  # Load onto device
        img = img.float().to(device)
        params = params.float().to(device)
        optimizer.zero_grad()  # set 0 grad
        output = model(img)
        loss = criterion(output, params)
        loss.backward()  # backpropogate weights
        optimizer.step()  
    return loss / len(train_loader.dataset)  # Divide by length of train-loader for relative loss


def test(model, test_loader, criterion, device, epoch):
    '''
    Test function for our model
    '''
    model.eval()  # Make sure to set to evaluation mode 
    test_loss = 0
    with torch.no_grad():  # No backpropogation during testing
        for i, (img, params) in enumerate(test_loader):
            img, params = img.to(device), params.to(device)
            img = img.float().to(device)
            params = params.float().to(device)
            output = model(img)
            loss = criterion(output, params)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader.dataset) # Divide by length of test-loader for relative loss
    print(f"Epoch {epoch}, Average test loss: {avg_test_loss}")    
    return avg_test_loss

