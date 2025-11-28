import torch.nn as nn


class DesPred(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim*4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim*4, config.output_dim)
        )

    def forward(self, x):
        x = self.mlp(x)

        return x
