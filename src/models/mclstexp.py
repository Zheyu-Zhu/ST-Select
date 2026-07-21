"""mclSTExp: Multimodal Contrastive Learning for ST (Briefings in Bioinformatics 2024)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .bleep import ImageEncoder, ExpressionEncoder


class MultiViewAugmentor(nn.Module):
    """Generate multiple views of the input for contrastive learning."""

    def __init__(self, embed_dim: int = 512, n_views: int = 3):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for _ in range(n_views)
        ])

    def forward(self, x: torch.Tensor) -> list:
        return [F.normalize(proj(x), dim=-1) for proj in self.projectors]


class MclSTExp(nn.Module):
    """
    BLEEP extended with multi-view contrastive augmentations.
    Multiple projection heads create diverse views for stronger contrastive signal.
    """

    def __init__(
        self,
        n_genes: int = 300,
        embed_dim: int = 512,
        temperature: float = 0.07,
        n_views: int = 3,
        pretrained: bool = True,
        retrieval_k: int = 50,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim, pretrained)
        self.expression_encoder = ExpressionEncoder(n_genes, embed_dim)
        self.img_augmentor = MultiViewAugmentor(embed_dim, n_views)
        self.expr_augmentor = MultiViewAugmentor(embed_dim, n_views)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.n_views = n_views
        self.retrieval_k = retrieval_k

        self.bank_embeddings: torch.Tensor = None
        self.bank_expressions: np.ndarray = None

    def forward(self, images: torch.Tensor, expressions: torch.Tensor = None):
        img_emb = self.image_encoder(images)

        if expressions is not None:
            expr_emb = self.expression_encoder(expressions)
            return img_emb, expr_emb

        return self.retrieve(img_emb)

    def multi_view_contrastive_loss(
        self, img_emb: torch.Tensor, expr_emb: torch.Tensor
    ) -> torch.Tensor:
        """Multi-view contrastive loss across all view pairs."""
        img_views = self.img_augmentor(img_emb)
        expr_views = self.expr_augmentor(expr_emb)

        total_loss = torch.tensor(0.0, device=img_emb.device)
        n_pairs = 0

        labels = torch.arange(len(img_emb), device=img_emb.device)

        for iv in img_views:
            for ev in expr_views:
                logits = (iv @ ev.T) / self.temperature
                loss_i2e = F.cross_entropy(logits, labels)
                loss_e2i = F.cross_entropy(logits.T, labels)
                total_loss += (loss_i2e + loss_e2i) / 2
                n_pairs += 1

        return total_loss / n_pairs

    def build_retrieval_bank(
        self,
        train_embeddings: np.ndarray,
        train_expressions: np.ndarray,
        train_indices=None,
        test_indices=None,
    ):
        """Build the retrieval bank from training data embeddings.

        The retrieval pool must contain ONLY training spots; including test spots
        leaks labels and inflates results (tutorial §6). Pass train_indices and
        test_indices to assert disjointness.
        """
        if train_indices is not None and test_indices is not None:
            overlap = set(map(int, train_indices)) & set(map(int, test_indices))
            if overlap:
                raise ValueError(
                    f"Retrieval bank leakage: {len(overlap)} spot(s) appear in "
                    f"both the train bank and the test set. The bank must contain "
                    f"only training spots."
                )
        self.bank_embeddings = torch.tensor(train_embeddings, dtype=torch.float32)
        self.bank_expressions = train_expressions

    def retrieve(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        if self.bank_embeddings is None:
            raise RuntimeError("Call build_retrieval_bank() first.")

        bank = self.bank_embeddings.to(query_embeddings.device)
        sim = query_embeddings @ bank.T
        _, topk_idx = sim.topk(self.retrieval_k, dim=-1)

        topk_idx_np = topk_idx.cpu().numpy()
        predictions = []
        for indices in topk_idx_np:
            pred = self.bank_expressions[indices].mean(axis=0)
            predictions.append(pred)

        return torch.tensor(np.array(predictions), dtype=torch.float32, device=query_embeddings.device)