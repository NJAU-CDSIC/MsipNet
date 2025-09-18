from __future__ import print_function
from tqdm import tqdm
import numpy as np
import torch
import utils.metrics as metrics


def trainstr(model, device, train_loader, criterion, optimizer, batch_size):
    """
    Trains the model for one epoch on structural data.
    Loads batches from train_loader and runs forward passes.
    Skips batches with only one class to ensure diversity.
    Computes loss, backpropagates, and updates model parameters.
    Tracks performance metrics using an MLMetrics object.
    """

    model.train()
    met = metrics.MLMetrics(objective='binary')
    for batch_idx, (x0, x00, y0) in enumerate(train_loader):
        x, h, y = x0.float().to(device), x00.float().to(device), y0.to(device).float()
        if y0.sum() == 0 or y0.sum() == batch_size:
            continue
        optimizer.zero_grad()
        output = model(x,h)
        loss = criterion(output, y)
        prob = torch.sigmoid(output)

        y_np = y.to(device='cpu', dtype=torch.long).detach().numpy()
        p_np = prob.to(device='cpu').detach().numpy()
        met.update(y_np, p_np, [loss.item()])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    return met

def validatestr(model, device, test_loader, criterion):
    """
    Evaluates the model on the given test dataset.
    Disables gradient computation to speed up inference.
    Collects true labels, predicted probabilities, and losses for all batches.
    Computes the mean loss across all batches.
    Updates and returns evaluation metrics using an MLMetrics object.
    """

    model.eval()
    y_all = []
    p_all = []
    l_all = []
    with torch.no_grad():
        for batch_idx, (x0, x00, y0) in enumerate(test_loader):
            x, h, y = x0.float().to(device), x00.float().to(device), y0.to(device).float()
            output = model(x,h)
            loss = criterion(output, y)
            prob = torch.sigmoid(output)

            y_np = y.to(device='cpu', dtype=torch.long).numpy()
            p_np = prob.to(device='cpu').numpy()
            l_np = loss.item()

            y_all.append(y_np)
            p_all.append(p_np)
            l_all.append(l_np)

    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    l_all = np.array(l_all)

    met = metrics.MLMetrics(objective='binary')
    met.update(y_all, p_all, [l_all.mean()])

    return met, y_all, p_all
