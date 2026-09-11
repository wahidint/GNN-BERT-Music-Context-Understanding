"""Multimodal BERT-GNN fusion models."""

import torch
import torch.nn as nn

class FusionClassifier(nn.Module):
    def __init__(self, bert_dim, gnn_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(bert_dim+gnn_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_embedding, graph_embedding):
        x = torch.cat([text_embedding, graph_embedding], dim=1)
        return self.fc(x)
