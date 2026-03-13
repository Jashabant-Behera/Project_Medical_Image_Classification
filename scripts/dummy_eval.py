import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ml.training.model import build_densenet121
from ml.training.evaluate import evaluate
 
os.makedirs('ml/saved_models', exist_ok=True)
model = build_densenet121(pretrained=True)
path = 'ml/saved_models/best_model_densenet121.pth'
torch.save({
    'epoch': 1,
    'model_state_dict': model.state_dict(),
    'val_auc': 0.5,
    'val_acc': 0.5,
    'model_name': 'densenet121'
}, path)
print(f"Dummy model saved to {path} for evaluation generation.")
evaluate(model_path=path)
