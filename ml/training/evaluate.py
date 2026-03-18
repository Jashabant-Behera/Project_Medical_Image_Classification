import os, sys, json
import torch, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ml.training.model import MODEL_REGISTRY
from ml.training.dataloader import get_dataloaders
 
def evaluate(model_path: str, model_name: str = 'densenet121',
             data_dir: str = 'ml/data/processed', report_dir: str = 'ml/reports'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(report_dir, exist_ok=True)
 
    model = MODEL_REGISTRY[model_name](pretrained=False).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
 
    num_workers = 4 if device.type != 'cpu' or sys.platform != 'win32' else 0
    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=32, num_workers=num_workers)
    all_preds, all_probs, all_labels = [], [], []
 
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
 
    # Metrics
    report = classification_report(all_labels, all_preds,
                                   target_names=['NORMAL','PNEUMONIA'], output_dict=True)
    auc = roc_auc_score(all_labels, all_probs)
    report['roc_auc'] = auc
    print('\n=== EVALUATION RESULTS ==='); print(classification_report(
        all_labels, all_preds, target_names=['NORMAL','PNEUMONIA']))
    print(f'ROC-AUC Score: {auc:.4f}')
 
    # Save metrics JSON
    with open(f'{report_dir}/metrics.json', 'w') as f: json.dump(report, f, indent=2)
 
    # Confusion Matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL','PNEUMONIA'], yticklabels=['NORMAL','PNEUMONIA'])
    plt.title('Confusion Matrix'); plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout(); plt.savefig(f'{report_dir}/confusion_matrix.png')
    # plt.show()
 
    # ROC Curve plot
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.4f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve'); plt.legend(); plt.tight_layout()
    plt.savefig(f'{report_dir}/roc_curve.png')
    # plt.show()
    print(f'Reports saved to {report_dir}/')
 
if __name__ == '__main__':
    evaluate(model_path='ml/saved_models/best_model_densenet121.pth')
