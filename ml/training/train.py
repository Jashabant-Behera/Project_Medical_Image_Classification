import os, sys, json, argparse, time
import torch, torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ml.training.model import MODEL_REGISTRY
from ml.training.dataloader import get_dataloaders
 
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc='Training', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total
 
@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for imgs, labels in tqdm(loader, desc='Evaluating', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Handle the case where there is only one class in the batch
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.5
    return total_loss / len(loader), correct / total, auc
 
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on: {device}')
    
    # Avoid multiprocessing issues on Windows by conditionally setting num_workers
    num_workers = 4 if device.type != 'cpu' or sys.platform != 'win32' else 0
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=num_workers)
        
    model = MODEL_REGISTRY[args.model](pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5)
 
    best_auc, patience_counter = 0.0, 0
    history = []
    os.makedirs(args.save_dir, exist_ok=True)
 
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = eval_epoch(model, val_loader, criterion, device)
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != prev_lr:
            print(f'  LR reduced: {prev_lr:.2e} → {new_lr:.2e}')
        elapsed = time.time() - t0
 
        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'TrainLoss={train_loss:.4f} Acc={train_acc:.4f} | '
              f'ValLoss={val_loss:.4f} Acc={val_acc:.4f} AUC={val_auc:.4f} | '
              f'Time={elapsed:.0f}s')
 
        history.append({'epoch': epoch, 'train_loss': train_loss,
                         'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': val_auc})
 
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            path = os.path.join(args.save_dir, f'best_model_{args.model}.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_auc': val_auc, 'val_acc': val_acc, 'model_name': args.model}, path)
            print(f'  ✓ Saved best model (AUC={best_auc:.4f}) -> {path}')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break
 
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Training complete. Best Val AUC: {best_auc:.4f}')
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',     default='densenet121')
    parser.add_argument('--epochs',    type=int, default=20)
    parser.add_argument('--batch_size',type=int, default=32)
    parser.add_argument('--lr',        type=float, default=1e-4)
    parser.add_argument('--patience',  type=int, default=5)
    parser.add_argument('--data_dir',  default='ml/data/processed')
    parser.add_argument('--save_dir',  default='ml/saved_models')
    main(parser.parse_args())
