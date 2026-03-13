import nbformat as nbf

nb = nbf.v4.new_notebook()

code1 = """# Cell 1 - Imports
import sys; sys.path.insert(0, '../..')
import torch
from PIL import Image
import matplotlib.pyplot as plt
from ml.training.model import MODEL_REGISTRY
from ml.training.augmentations import inference_transform
from backend.services.gradcam import GradCAMService"""

code2 = """# Cell 2 - Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MODEL_REGISTRY['densenet121'](pretrained=False).to(device)
ckpt = torch.load('../../ml/saved_models/best_model_densenet121.pth', map_location=device, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()"""

code3 = """# Cell 3 - Test on a PNEUMONIA sample
import glob
# try to find a sample dynamically to avoid file not found
img_path_candidates = glob.glob('../../ml/data/raw/test/PNEUMONIA/*.jpeg')
if len(img_path_candidates) == 0:
    print('No image found')
else:
    img_path = img_path_candidates[0]
    original = Image.open(img_path).convert('RGB')
    tensor = inference_transform(original).unsqueeze(0).to(device)"""

code4 = """# Cell 4 - Generate prediction + Grad-CAM
gradcam = GradCAMService(model, target_layer_name='features.norm5')
with torch.no_grad():
    # GradCAM needs gradients, so we temporarily enable gradients for input if we need to?
    # Actually wait. the service enables backward by running output.backward() so we don't want no_grad.
    pass

# Remove no_grad because GradCAM does a backward pass
tensor.requires_grad = True
out = model(tensor)
probs = torch.softmax(out, dim=1)
class_idx = out.argmax(dim=1).item()
confidence = probs[0][class_idx].item()
 
class_names = ['NORMAL', 'PNEUMONIA']
print(f'Prediction: {class_names[class_idx]} (Confidence: {confidence:.4f})')"""

code5 = """# Cell 5 - Visualize overlay
overlay = gradcam.generate_overlay(tensor, original, class_idx)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(original, cmap='gray'); axes[0].set_title('Original X-Ray'); axes[0].axis('off')
axes[1].imshow(overlay); axes[1].set_title(f'Grad-CAM: {class_names[class_idx]} ({confidence:.2%})')
axes[1].axis('off')
plt.tight_layout(); plt.savefig('../../ml/reports/gradcam_sample.png')
# plt.show()
print('GradCAM saved generated successfully.')"""

nb['cells'] = [
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_code_cell(code5),
]

with open('ml/notebooks/04_gradcam_test.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Grad-CAM notebook created successfully.")
