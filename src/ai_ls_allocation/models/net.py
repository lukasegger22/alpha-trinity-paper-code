import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1, output_dim=1):
        super(GRUModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU Layer
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass
        # out shape: (batch_size, seq_len, hidden_dim)
        out, _ = self.gru(x, h0)
        
        # WICHTIGE ÄNDERUNG:
        # Wir schicken jetzt ALLE Zeitschritte durch den Linear Layer, 
        # nicht nur den letzten. Damit bekommen wir eine Vorhersage für JEDEN Tag.
        prediction = self.fc(out)
        
        return prediction