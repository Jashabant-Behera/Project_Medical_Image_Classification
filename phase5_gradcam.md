# Phase 5: Grad-CAM Explainability

## Objective
Implement Class Activation Mapping (Grad-CAM) visualizations into the classification pipelines dynamically capturing neural network logic (activation maps/gradients layers natively inside PyTorch) generating `numpy.ndarray` color overlays to verify what region of an X-Ray drives a specific diagnosis structurally. Translating PyTorch hook implementations logically to bypass Native Tensor in-place mutations (like `ReLU`).

## Files Created
1. `backend/services/gradcam.py` - Core PyTorch interaction `GradCAMService(model, 'features.norm5')`. Dynamically registers `forward_hooks` capturing model inferences natively and then attaching direct `tensor.register_hook(backward_gradient_saves)` to bypass inplace failures correctly. Performs dynamic spatial pooling translating raw matrices (`[2,3] dim`) down into normalized output maps. Overrides raw arrays into `cv2.applyColorMap(.., cv2.COLORMAP_JET)` interpolations wrapping Native image structures dynamically to return `Image.fromarray(overlay)` encoding Base64 sequences seamlessly.
2. `ml/notebooks/04_gradcam_test.ipynb` - Smoke testing suite evaluating Grad-CAM capabilities over randomized test images interactively. Verify dynamic generation capabilities `Prediction -> Matrix Mapping -> Heatmap` and saves `ml/reports/gradcam_sample.png`.

## Core Logic Fix Action
Initial attempts to register full backward hooks directly onto DenseNet blocks failed due to PyTorch `inplace=True` Relu overrides masking out gradients natively. This was fully overridden in the pipeline via direct activation polling `output.requires_grad: -> output.register_hook(self._save_gradient)`. Explainability flows securely now.
