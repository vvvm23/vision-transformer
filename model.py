import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, nb_in, dropout=0.1, max_length=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, nb_in)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, nb_in, 2).float() * (-torch.log(torch.tensor(10000.0)) / nb_in))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# VisionTransformer module
class VisionTransformer(nn.Module):
    def __init__(self, img_dim, patch_dim, out_dim, nb_channels, emd_dim, nb_heads, nb_layers, h_dim, dropout=0.1):
        super().__init__()

        assert(emd_dim % nb_heads == 0) # Allows for splitting dimensions between attention heads
        assert(img_dim % patch_dim == 0) # For simplicity, enforce that patches must be the same size
        
        self.emd_dim = emd_dim
        self.nb_heads = nb_heads
        self.patch_dim = patch_dim
        self.nb_channels = nb_channels

        self.nb_patches = int((img_dim // patch_dim) ** 2) 
        self.flatten_dim = patch_dim * patch_dim * nb_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, emd_dim)
        self.position_encoding = PositionalEncoding(emd_dim, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(emd_dim, nb_heads, h_dim, dropout=dropout)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, nb_layers)

        self.decoder = nn.Sequential(
            nn.Linear(emd_dim*self.nb_patches, h_dim),
            nn.ReLU(inplace = True),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, x):
        # Perform patch and flatten operations
        x = x.unfold(2, self.patch_dim, self.patch_dim).unfold(3, self.patch_dim, self.patch_dim).contiguous()
        x = x.view(x.size(0), -1, self.flatten_dim)
        # x = x.permute(1, 0, 2) # do we need to permute?

        x = F.relu(self.linear_encoding(x)) # Identically apply an encoding to all flattened patches
        x = self.position_encoding(x) # TODO: verify positional encoding is working as intended

        x = self.trans_encoder(x) # Pass patch sequence to the Transformer Encoder

        # Decode and output class probabilities
        x = x.view(-1, self.emd_dim*self.nb_patches)
        x = self.decoder(x)
        x = F.log_softmax(x, dim=-1)

        return x

if __name__ == '__main__':
    model = VisionTransformer(28, 7, 10, 1, 256, 8, 6, 1024)
    print(model)

    x = torch.randn(8, 1, 28, 28)
    print(model(x))
