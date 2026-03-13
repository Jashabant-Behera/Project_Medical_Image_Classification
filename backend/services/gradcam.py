import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import base64, io
 
class GradCAMService:
    '''
    Generates Grad-CAM heatmap overlays for DenseNet121.
    Uses PyTorch hooks to capture gradients and activations
    from the last convolutional layer (features.norm5).
    '''
    def __init__(self, model: torch.nn.Module, target_layer_name: str = 'features.norm5'):
        self.model = model
        self.gradients = None
        self.activations = None
        self.target_layer_name = target_layer_name
        # Register hooks on target layer
        target_layer = self._get_layer(target_layer_name)
        # Using a forward hook to capture activations AND register a tensor hook
        # for gradients, which perfectly bypasses the inplace ReLU bug in DenseNet!
        target_layer.register_forward_hook(self._save_activation_and_gradient)
 
    def _get_layer(self, layer_name: str):
        parts = layer_name.split('.')
        layer = self.model
        for part in parts:
            layer = getattr(layer, part)
        return layer
 
    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def _save_activation_and_gradient(self, module, input, output):
        self.activations = output.detach()
        # Registering the backward hook directly on the resulting tensor 
        if output.requires_grad:
            output.register_hook(self._save_gradient)
 
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        '''Generate CAM for a given class index. Returns heatmap in [0,1].'''
        self.model.eval()
        
        # Zero gradients
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Backprop on predicted class score
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot)
        
        # Pool gradients across channels
        pooled_grads = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (pooled_grads * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
 
    def generate_overlay(self, input_tensor: torch.Tensor,
                         original_image: Image.Image,
                         class_idx: int) -> Image.Image:
        '''Returns PIL Image with Grad-CAM heatmap blended over original.'''
        cam = self.generate_cam(input_tensor, class_idx)
        # Resize CAM to original image size
        h, w = original_image.size[1], original_image.size[0]
        cam_resized = cv2.resize(cam, (w, h))
        # Apply jet colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        # Blend with original image
        original_np = np.array(original_image.convert('RGB'))
        overlay = cv2.addWeighted(original_np, 0.5, heatmap, 0.5, 0)
        return Image.fromarray(overlay)
 
    def overlay_to_base64(self, overlay_image: Image.Image) -> str:
        '''Convert PIL overlay image to base64 PNG string for API response.'''
        buffer = io.BytesIO()
        overlay_image.save(buffer, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
