# Phase 3: Model Development — Transfer Learning with DenseNet121

## Objective
Initiate the deep-learning Transfer Layer structures parsing baseline models (DenseNet, ResNet, EfficientNet) via PyTorch integrations, downloading pretrained `IMAGENET1K` metadata sequentially to bootstrap analytical operations cleanly into localized nodes.

## Files Created
1. `ml/training/model.py` - Core PyTorch `torch.nn.Module` configurations holding the `build_densenet121()` mapping logic. Contains dynamic conditional mappings for overriding standard architectures (`freeze_layers=True`). Freezes `param.requires_grad = False` throughout earlier neural structures selectively activating `features.denseblock4` backpropagation for edge refinement and creating a bespoke binary sequential classifier (`nn.Linear(1024, 256) -> ReLU -> Dropout(0.4) -> Linear(2)`).
2. `ml/notebooks/03_transfer_learning.ipynb` - Jupyter node interactive experiment space designed explicitly for analyzing model shapes, dynamically unfreezing node components selectively, measuring 1-epoch runs over training loaders interactively before executing large-scale pipelines, confirming Trainable Parameter ratios (`2,423,042` / `~8.0M`).

## Methodologies
**DenseNet121** architectures natively utilize deep concatenations enabling exceptional gradient flows scaling back from upper hierarchical classification levels down through root input matrices natively; this naturally avoids traditional localized "vanishing gradient" fallouts typical in standardized residual layers while conserving extensive memory overhead for the local `CPU` training loops.
