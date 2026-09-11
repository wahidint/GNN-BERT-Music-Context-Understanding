"""Training entry point.

Training loops from the experiment notebook are organized here.
"""

def train_model(model, loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
