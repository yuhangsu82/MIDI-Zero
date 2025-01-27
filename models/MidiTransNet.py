import torch
import torch.nn as nn
import torch.nn.functional as F


class MidiTransNet(nn.Module):
    def __init__(self, vocab_size=90, embed_dim=128, num_heads=8, num_layers=4, max_len=256, dropout=0.1, pitch_only=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.pitch_only = pitch_only
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        if not self.pitch_only:
            self.time_embedding = nn.Linear(1, embed_dim)

        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='relu',
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.output_projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, time_info=None):
        token_embed = self.token_embedding(x)
        
        if not self.pitch_only:
            time_embed = self.time_embedding(time_info.unsqueeze(-1))
            mask = (time_info != 0).float()
            time_embed = time_embed * mask.unsqueeze(-1)
            combined_embed = token_embed + time_embed
        else:
            combined_embed = token_embed
        
        combined_embed += self.positional_encoding[:, :x.size(1), :]
        transformer_output = self.transformer(combined_embed)
        sequence_output = transformer_output.mean(dim=1)
        embedding = self.output_projection(sequence_output)

        return embedding


if __name__ == '__main__':
    model = MidiTransNet()  # Initialize the model
    x = torch.randint(0, 90, (2, 5))  # MIDI token sequence for 2 batches
    print(x.shape)
    time_info = torch.tensor([[0.1, 0.2, 0.0, 0.0, 0.0], [0.3, 0.0, 0.4, 0.0, 0.0]]) # Time information for 2 batches
    output = model(x, time_info)
    print(output.shape)
