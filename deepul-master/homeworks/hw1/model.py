import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask_type='A', **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, H, W = self.weight.shape

        # create mask
        self.mask[:, :, H//2, W//2:] = 0
        self.mask[:, :, H//2, :] = 0
        if mask_type == 'A':
            self.mask[:, :, H//2, W//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
    
class PixelCNN(nn.Module):
    def __init__(self, input_channels=1, n_filters=64):
        super().__init__()

        # First layer - 7x7 masked type A convolution
        self.conv1 = MaskedConv2d(
            input_channels, n_filters, 
            kernel_size=7, padding=3,
            mask_type='A'
        )

        # 5 layers of 7x7 masked type B convolutions
        self.hidden_layers = nn.ModuleList([
            MaskedConv2d(
                n_filters, n_filters,
                kernel_size=7, padding=3,
                mask_type='B'
            ) for _ in range(5)
        ])

        # 2 layers of 1x1 masked type B convolutions
        self.final_layers = nn.ModuleList([
            MaskedConv2d(
                n_filters, n_filters,
                kernel_size=1,  # 1x1 convolution
                mask_type='B'
            ) for _ in range(2)
        ])

        # Output layer
        self.output = nn.Conv2d(n_filters, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        for layer in self.final_layers:
            x = F.relu(layer(x))

        return self.output(x)