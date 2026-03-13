import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ml.training.model import MODEL_REGISTRY
from backend.config.settings import settings
 
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
 
class InferenceService:
    _instance = None
 
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance
 
    def load_model(self):
        if self._loaded: return
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MODEL_REGISTRY[settings.MODEL_NAME](pretrained=False).to(self.device)
        ckpt = torch.load(settings.MODEL_PATH, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self._loaded = True
        print(f'Model loaded on {self.device}: {settings.MODEL_NAME} v{settings.MODEL_VERSION}')
 
    def predict(self, tensor: torch.Tensor) -> tuple[str, float, int]:
        '''Returns (class_name, confidence, class_index).'''
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            class_idx = probs.argmax(dim=1).item()
            confidence = probs[0][class_idx].item()
        return CLASS_NAMES[class_idx], confidence, class_idx
 
inference_service = InferenceService()
