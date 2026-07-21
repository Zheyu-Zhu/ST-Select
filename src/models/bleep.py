"""BLEEP: Bi-modal Contrastive Learning for ST Expression Prediction (NeurIPS 2024)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class ImageEncoder(nn.Module):
    """DenseNet-121 image encoder for BLEEP."""

    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        densenet = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
        )
        self.features = nn.Sequential(*list(densenet.features.children()))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1024, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = self.pool(f).flatten(1)
        return F.normalize(self.proj(f), dim=-1)


class ExpressionEncoder(nn.Module):
    """MLP encoder for log-normalized expression vectors."""

    def __init__(self, n_genes: int = 300, embed_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class BLEEP(nn.Module):
    """
    CLIP-style bi-modal contrastive model.
    At inference: retrieve top-k training expressions by image embedding similarity,
    then average them as the prediction.
    """

    def __init__(
        self,
        n_genes: int = 300,
        embed_dim: int = 512,
        temperature: float = 0.07,
        pretrained: bool = True,
        retrieval_k: int = 50,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim, pretrained)
        self.expression_encoder = ExpressionEncoder(n_genes, embed_dim)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.retrieval_k = retrieval_k

        # Retrieval bank (populated from training data)
        self.bank_embeddings: torch.Tensor = None
        self.bank_expressions: np.ndarray = None

    def forward(self, images: torch.Tensor, expressions: torch.Tensor = None):
        """
        Training: return contrastive loss logits.
        Inference (expressions=None): return retrieved expression predictions.
        """
        img_emb = self.image_encoder(images)

        if expressions is not None:
            expr_emb = self.expression_encoder(expressions)
            return img_emb, expr_emb

        # Inference via retrieval
        return self.retrieve(img_emb)

    def contrastive_loss(self, img_emb: torch.Tensor, expr_emb: torch.Tensor) -> torch.Tensor:
        """Symmetric CLIP loss."""
        logits = (img_emb @ expr_emb.T) / self.temperature
        labels = torch.arange(len(logits), device=logits.device)
        loss_i2e = F.cross_entropy(logits, labels)
        loss_e2i = F.cross_entropy(logits.T, labels)
        return (loss_i2e + loss_e2i) / 2

    def build_retrieval_bank(
        self,
        train_embeddings: np.ndarray,
        train_expressions: np.ndarray,
        train_indices=None,
        test_indices=None,
    ):
        """Build the retrieval bank from training data embeddings.

        The retrieval pool must contain ONLY training spots — including any
        validation/test spots leaks labels and inflates results (tutorial §6).
        Pass train_indices and test_indices to assert this invariant.
        """
        if train_indices is not None and test_indices is not None:
            overlap = set(map(int, train_indices)) & set(map(int, test_indices))
            if overlap:
                raise ValueError(
                    f"Retrieval bank leakage: {len(overlap)} spot(s) appear in "
                    f"both the train bank and the test set (e.g. {sorted(overlap)[:5]}). "
                    f"The bank must contain only training spots."
                )
        self.bank_embeddings = torch.tensor(train_embeddings, dtype=torch.float32)
        self.bank_expressions = train_expressions

    def retrieve(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        """Retrieve and average top-k expressions."""
        if self.bank_embeddings is None:
            raise RuntimeError("Call build_retrieval_bank() first.")

        bank = self.bank_embeddings.to(query_embeddings.device)
        sim = query_embeddings @ bank.T  # (B, N_bank)
        _, topk_idx = sim.topk(self.retrieval_k, dim=-1)  # (B, K)

        # Average retrieved expressions
        topk_idx_np = topk_idx.cpu().numpy()
        predictions = []
        for indices in topk_idx_np:
            pred = self.bank_expressions[indices].mean(axis=0)
            predictions.append(pred)

        return torch.tensor(np.array(predictions), dtype=torch.float32, device=query_embeddings.device)