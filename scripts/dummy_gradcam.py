import os, sys, torch
from PIL import Image
import matplotlib.pyplot as plt
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.training.model import MODEL_REGISTRY
from ml.training.augmentations import inference_transform
from backend.services.gradcam import GradCAMService

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MODEL_REGISTRY['densenet121'](pretrained=False).to(device)
ckpt = torch.load('ml/saved_models/best_model_densenet121.pth', map_location=device, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

img_path_candidates = glob.glob('ml/data/raw/test/PNEUMONIA/*.jpeg')
if len(img_path_candidates) == 0:
    print('No image found')
    sys.exit(0)
img_path = img_path_candidates[0]
original = Image.open(img_path).convert('RGB')
tensor = inference_transform(original).unsqueeze(0).to(device)

gradcam = GradCAMService(model, target_layer_name='features.norm5')

tensor.requires_grad = True
out = model(tensor)
probs = torch.softmax(out, dim=1)
class_idx = out.argmax(dim=1).item()
confidence = probs[0][class_idx].item()

class_names = ['NORMAL', 'PNEUMONIA']
print(f'Prediction: {class_names[class_idx]} (Confidence: {confidence:.4f})')

overlay = gradcam.generate_overlay(tensor, original, class_idx)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(original, cmap='gray'); axes[0].set_title('Original X-Ray'); axes[0].axis('off')
axes[1].imshow(overlay); axes[1].set_title(f'Grad-CAM: {class_names[class_idx]} ({confidence:.2%})')
axes[1].axis('off')
plt.tight_layout(); plt.savefig('ml/reports/gradcam_sample.png')
print('GradCAM saved generated successfully.')
