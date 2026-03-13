import torch
from PIL import Image
from torchvision import transforms
 
class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
 
    def process(self, image: Image.Image) -> torch.Tensor:
        '''Convert PIL Image -> normalized tensor with batch dim [1, 3, 224, 224].'''
        return self.transform(image.convert('RGB')).unsqueeze(0)
