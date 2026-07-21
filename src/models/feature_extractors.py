"""Feature extractors: CONCH, UNI, DINOv2, etc. for frozen backbone features."""

from typing import Optional

import numpy as np
import torch
from ..utils.reproducibility import resolve_device as _resolve_device
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class FeatureExtractor:
    """Base class for frozen feature extraction from H&E patches."""

    def __init__(self, model: nn.Module, transform: T.Compose, device: str = "cuda"):
        self.model = model.eval()
        self.transform = transform
        self.device = _resolve_device(device)
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor) -> np.ndarray:
        images = images.to(self.device)
        features = self.model(images)
        return features.cpu().numpy()

    @torch.no_grad()
    def extract_from_dataloader(self, dataloader, key: str = "image") -> np.ndarray:
        all_features = []
        for batch in dataloader:
            images = batch[key] if isinstance(batch, dict) else batch[0]
            images = images.to(self.device)
            features = self.model(images)
            all_features.append(features.cpu().numpy())
        return np.concatenate(all_features, axis=0)


def get_feature_extractor(name: str, device: str = "cuda") -> FeatureExtractor:
    """Factory for common pathology feature extractors."""
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if name == "densenet121":
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model = nn.Sequential(
            densenet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        return FeatureExtractor(model, transform, device)

    elif name == "resnet50":
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        return FeatureExtractor(model, transform, device)

    elif name == "vit_b_16":
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # Use CLS token
        class ViTFeatureExtractor(nn.Module):
            def __init__(self, vit):
                super().__init__()
                self.vit = vit
            def forward(self, x):
                x = self.vit._process_input(x)
                n = x.shape[0]
                cls_token = self.vit.class_token.expand(n, -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                x = self.vit.encoder(x)
                return x[:, 0]

        model = ViTFeatureExtractor(vit)
        return FeatureExtractor(model, transform, device)

    elif name == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        return FeatureExtractor(model, transform, device)

    elif name == "conch":
        # CONCH requires the official repo to be installed
        try:
            from conch.open_clip_custom import create_model_from_pretrained
            model, preprocess = create_model_from_pretrained("conch_ViT-B-16")
            class CONCHWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def forward(self, x):
                    return self.model.encode_image(x)
            return FeatureExtractor(CONCHWrapper(model), preprocess, device)
        except ImportError:
            raise ImportError("Install CONCH: pip install git+https://github.com/mahmoodlab/CONCH.git")

    elif name == "uni":
        try:
            import timm
        except ImportError:
            raise ImportError("UNI needs timm + huggingface_hub: pip install 'timm>=0.9'")
        # UNI (Mahmood Lab, Nat. Med. 2024): ViT-L/16, gated on HuggingFace.
        # init_values + dynamic_img_size are required to build the correct
        # architecture for the released weights; the repo is gated, so this
        # will raise a clear GatedRepoError if the user hasn't been granted
        # access / logged in (huggingface-cli login). UNI expects ImageNet
        # normalization, which `transform` already applies.
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI", pretrained=True,
            init_values=1e-5, dynamic_img_size=True,
        )
        return FeatureExtractor(model, transform, device)

    else:
        raise ValueError(f"Unknown feature extractor: {name}. "
                        f"Available: densenet121, resnet50, vit_b_16, dinov2, conch, uni")
