import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from datasets_elliptic.old.utils import accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import precision_recall_fscore_support, f1_score

def train(args, model, data):
    """Train a GNN ethident and return the trained ethident."""
    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.9, verbose=True)
    epochs = args['epochs']
    model.train()

    best_val_loss = float('inf')
    patience = 1000
    epochs_since_best = 0

    train_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.4, 0.6])).to(args['device'])

    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        out, _ = model((data.x, data.edge_index))
        # loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss = train_loss(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
          val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])

          prec_ill, rec_ill, f1_ill, _ = precision_recall_fscore_support(data.y[data.val_mask].to("cpu"),
                                                                         out[data.val_mask].argmax(dim=1).to("cpu"), average='binary',
                                                                         pos_label=0)
          f1_micro = f1_score(data.y[data.val_mask].to("cpu"), out[data.val_mask].argmax(dim=1).to("cpu"), average='micro')

          val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

          # Adjust learning rate
          scheduler.step(val_loss)

        # Check if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # Print metrics every 10 epochs
        if epoch % 1 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  f'{acc*100:>6.2f}% | Val Loss: {val_loss:.3f} | '
                  f'Val Acc: {val_acc*100:.2f}%'
                  f' | Val F1: {f1_micro:.3f}'
                  f' | F1 Illicit: {f1_ill:.3f}'
                  f' | Recall: {rec_ill:.3f}'
                  f' | Precision: {prec_ill:.3f}'
                  )

        # Check if early stopping criteria is met
        if epochs_since_best >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    return model

@torch.no_grad()
def test(model, data):
    """Evaluate the ethident on test set and print the accuracy score."""
    model.eval()
    out, _ = model((data.x, data.edge_index))
    acc = accuracy(out[data.test_mask].argmax(dim=1), data.y[data.test_mask])
    return acc