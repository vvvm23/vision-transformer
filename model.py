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
        x = F.softmax(x, dim=-1)

        return x

# Baseline module 
class ResidualBlock(nn.Module):
    def __init__(self, nb_channels):
        super().__init__()
        self.nb_channels = nb_channels
        self.conv = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(nb_channels)

        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.bn.weight, 0.5)
        torch.nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        z = F.relu(self.bn(self.conv(x)))
        return x + z

class BaseModel(nn.Module):
    def __init__(self, img_dim, nb_channels, out_dim, res_channels=32, nb_res_blocks=10, mlp_dim=32):
        super().__init__()
        self.nb_channels = nb_channels
        self.img_dim = img_dim
        self.flatten_dim = res_channels*(img_dim // 4)**2

        self.in_conv = nn.Conv2d(nb_channels, res_channels, kernel_size=3, padding=1)
        self.res_layers = nn.Sequential(*(nb_res_blocks*[ResidualBlock(res_channels)]))
        self.mlp1 = nn.Linear(self.flatten_dim, mlp_dim)
        self.mlp2 = nn.Linear(mlp_dim, out_dim)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.in_conv(x)), 2)
        x = self.res_layers(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.flatten_dim)
        x = F.relu(self.mlp1(x))
        x = self.mlp2(x)
        return F.softmax(x, dim=-1)

if __name__ == '__main__':
    # model = VisionTransformer(28, 7, 10, 1, 256, 8, 6, 1024)
    # print(model)

    # x = torch.randn(8, 1, 28, 28)
    # print(model(x))
    model = BaseModel(32, 3, 10)
    x = torch.randn(1, 3, 32, 32)
    print(model)
    print(model(x))
