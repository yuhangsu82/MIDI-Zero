import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlledMusicEmbeddingLoss(nn.Module):
    """
    Combined loss function for music embedding training, integrating InfoNCE
    with batch-based negatives and controlled augmentation similarity.
    """
    def __init__(self, tau=0.1, alpha=0.5, loss_mode='infonce'):
        super(ControlledMusicEmbeddingLoss, self).__init__()
        self.tau = tau  # Temperature parameter for InfoNCE
        self.alpha = alpha
        self.loss_mode = loss_mode

    def forward(self, anchor, positives, refs):
        """
        Args:
            anchor: Tensor of shape (batch_size, embed_dim), embeddings for the anchor (original samples).
            positives: Tensor of shape (batch_size, embed_dim), embeddings for augmented samples.
            refs: Tensor of shape (batch_size), similarity reference values for each positive.

        Returns:
            loss: Combined loss value.
        """
        anchor = F.normalize(anchor, dim=-1)  # (batch_size, embed_dim)
        positives = F.normalize(positives, dim=-1)  # (batch_size, embed_dim)
        batch_size = anchor.size(0)
        similarity_matrix = torch.matmul(anchor, positives.T)  # Shape: (batch_size, batch_size)
        similarity_matrix /= self.tau  # Scale by temperature
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        loss_infonce = F.cross_entropy(similarity_matrix, labels)
        sim_pos = (anchor * positives).sum(dim=-1)  # Cosine similarity for positive pairs
        loss_aug = ((sim_pos - refs) ** 2).mean()

        if self.loss_mode == 'infonce':
            return self.alpha * loss_infonce
        elif self.loss_mode == 'sim':
            return (1 - self.alpha) * loss_aug
        loss = (1 - self.alpha) * loss_aug + self.alpha * loss_infonce

        return loss
