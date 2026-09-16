import torch
from tqdm import trange


def train(objective, parameters, num_steps=5000, learning_rate=0.01, decay=0.8, decay_every=500,
          sync_every=50):
    """The standard loop: objective() returns the loss, parameters are whatever it should move.

    parameters is materialised with list() because model.parameters() is a generator: reusing
    one across two train() calls used to hand Adam an empty list on the second call.
    Losses are read back every sync_every steps in one transfer; loss.item() on every step
    is a device sync per step, which on a GPU/MPS stalls the whole pipeline.
    """
    parameters = list(parameters)
    optimiser = torch.optim.Adam(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, decay_every, decay)

    losses, pending = [], []
    for _ in trange(num_steps, desc="Training"):
        optimiser.zero_grad()
        loss = objective()
        loss.backward()
        optimiser.step()
        scheduler.step()
        pending.append(loss.detach())
        if len(pending) == sync_every:
            losses += torch.stack(pending).tolist()
            pending = []
    if pending:
        losses += torch.stack(pending).tolist()
    return losses
