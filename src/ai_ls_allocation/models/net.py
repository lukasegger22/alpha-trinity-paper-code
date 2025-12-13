import torch
import torch.nn as nn

class QuantileGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        
        # Das GRU-Layer verarbeitet die Zeitreihe
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Der Output-Head transformiert den hidden state in 3 Quantile
        self.head = nn.Linear(hidden_size, 3) 

    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        out, _ = self.gru(x)
        
        # Wir nehmen nur den Output des letzten Zeitschritts
        last_step = out[:, -1, :]
        
        # Vorhersage: (batch_size, 3) -> [q10, q50, q90]
        quantiles = self.head(last_step)
        return quantiles

    def predict_sorted(self, x):
        """
        Hilfsfunktion für Inference:
        Stellt sicher, dass Q10 <= Q50 <= Q90 ist (Crossing Problem).
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            # Sortieren entlang der letzten Dimension (axis 1)
            sorted_out, _ = torch.sort(out, dim=1)
        return sorted_out

def quantile_loss(preds, target, quantiles=[0.1, 0.5, 0.9]):
    """
    Pinball Loss (Quantile Loss) für neuronale Netze.
    preds: (batch, 3)
    target: (batch, 1)
    """
    losses = []
    for i, q in enumerate(quantiles):
        error = target - preds[:, i].unsqueeze(1)
        loss = torch.max((q - 1) * error, q * error)
        losses.append(loss)
        
    return torch.mean(torch.sum(torch.stack(losses, dim=1), dim=1))