"""Train per-horizon models; choose checkpoints only by validation loss."""
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .model import TrafficGRU


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def fit(train, validation, config, seed):
    seed_all(seed)
    model = TrafficGRU(train[0].shape[-1], config['hidden_size'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(*(torch.from_numpy(a) for a in train)),
                        batch_size=config['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    vx, vy = (torch.from_numpy(a) for a in validation)
    best, best_state, stale, history = float('inf'), None, 0, []
    for epoch in range(config['epochs']):
        model.train()
        loss_sum = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(y)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(vx), vy).item()
        history.append({'epoch': epoch+1, 'train_mse_scaled': loss_sum/len(train[1]), 'validation_mse_scaled': val_loss})
        if (epoch + 1) % 10 == 0:
            print(f'  seed={seed} epoch={epoch+1} validation_mse={val_loss:.6f}', flush=True)
        if not np.isfinite(val_loss):
            raise ValueError('Non-finite validation loss')
        if val_loss < best:
            best, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= config['patience']:
            break
    model.load_state_dict(best_state)
    return model, history


def predict(model, x, scaler):
    model.eval()
    with torch.no_grad():
        values = torch.cat([model(chunk) for chunk in torch.from_numpy(x).split(512)]).numpy()
    return values * scaler['std'] + scaler['mean']


def metrics(actual, predicted):
    error = actual - predicted
    denom = np.sum((actual-actual.mean())**2)
    return {'r2': float(1 - np.sum(error**2)/denom) if len(actual) >= 2 and denom > 0 else None,
            'rmse': float(np.sqrt(np.mean(error**2))), 'mae': float(np.mean(np.abs(error)))}
