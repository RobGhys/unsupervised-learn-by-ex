import torch
import torch.nn as nn


class DepthToSpace(nn.Module):
    """
    This module reorganizes depth/channel information into spatial dimensions.
    Used for upsampling where we trade channel depth for spatial resolution.
    
    If block_size = 2:
    INPUT:  [batch, C*4, H, W]      # More channels, smaller spatial dims
    OUTPUT: [batch, C, H*2, W*2]    # Fewer channels, larger spatial dims
    """
    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size**2

   
    def forward(self, input):
        # input shape: [batch, channels, height, width]
        output = input.permute(0, 2, 3, 1)  
        # shape: [batch, height, width, channels]
        
        (batch_size, d_height, d_width, d_depth) = output.size()
        # Calculate new dimensions:
        # s_depth = d_depth/(block_size*block_size) -> new channel depth
        # s_width = d_width * block_size -> new wider width
        # s_height = d_height * block_size -> new taller height
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        
        # Reshape to prepare for spatial reorganization
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        # shape: [batch, height, width, block_size*block_size, new_depth]
        
        # Split along the block_size_sq dimension
        spl = t_1.split(self.block_size, 3)
        # Creates list of tensors, each split along 4th dim
        
        # Reshape each split piece
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        
        # Complex reshaping operations to properly arrange spatial dimensions
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(
            batch_size, s_height, s_width, s_depth)
        
        # Final permute to get back to PyTorch's standard BCHW format
        output = output.permute(0, 3, 1, 2)
        # Final shape: [batch, new_channels, new_height, new_width]
        return output


class SpaceToDepth(nn.Module):
   """
   This module does the opposite of DepthToSpace - it reorganizes spatial information into depth/channels.
   Used for downsampling where we trade spatial resolution for channel depth.
   
   If block_size = 2:
   INPUT:  [batch, C, H*2, W*2]    # Fewer channels, larger spatial dims
   OUTPUT: [batch, C*4, H, W]      # More channels, smaller spatial dims
   """
   def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size**2

   def forward(self, input):
       # input shape: [batch, channels, height, width]
       output = input.permute(0, 2, 3, 1)
       # shape: [batch, height, width, channels]
       
       (batch_size, s_height, s_width, s_depth) = output.size()
       # Calculate new dimensions:
       # d_depth = s_depth * (block_size*block_size) -> more channels
       # d_width = s_width/block_size -> smaller width
       # d_height = s_height/block_size -> smaller height
       d_depth = s_depth * self.block_size_sq
       d_width = int(s_width / self.block_size)
       d_height = int(s_height / self.block_size)
       
       # Split along width dimension
       t_1 = output.split(self.block_size, 2)
       # Creates list of tensors split along width
       
       # Reshape each split piece, collapsing spatial dims into channels
       stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
       
       # Stack and permute operations to get proper arrangement
       output = torch.stack(stack, 1)
       output = output.permute(0, 2, 1, 3)
       output = output.permute(0, 3, 1, 2)
       # Final shape: [batch, new_channels, new_height, new_width]
       return output


# Spatial Upsampling with Nearest Neighbors
class Upsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.depth_to_space = DepthToSpace(block_size=2)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=1)
        x = self.depth_to_space(x)
        x = self.conv(x)

        return x     


# Spatial Downsampling with Spatial Mean Pooling
class Downsample_Conv2d(nn.Module):
        def __init__(self, in_dim, out_dim, bias=False, kernel_size=(3, 3), stride=1, padding=1):
            super().__init__()
            self.space_to_depth = SpaceToDepth(2)
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        
        def forward(self, x):
            x = self.space_to_depth(x)
            chunks = x.chunk(4, dim=1)
            x = torch.stack(chunks, dim=0).sum(dim=0) / 4.0
            x = self.conv(x)

            return x


class ResnetBlockUp(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.batch_norm_1 = nn.BatchNorm2d(in_dim)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_dim, n_filters, kernel_size, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(n_filters)
        self.upsample_conv_1 = Upsample_Conv2d(n_filters, n_filters, kernel_size, padding=1)
        self.upsample_conv_2 = Upsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        _x = x
        _x = self.batch_norm_1(_x)
        _x = self.relu(_x)
        _x = self.conv(_x)
        _x = self.batch_norm_2(_x)
        _x = nn.ReLU()(_x)
        residual = self.upsample_conv_1(_x)
        shortcut = self.upsample_conv_2(x)

        return residual + shortcut


#The ResBlockDown module is similar, except it uses Downsample_Conv2d and omits the BatchNorm.
class ResBlockDown(nn.Module):
    def __init__(self, in_dim, bias=False, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_dim, n_filters, kernel_size, padding=1)
        self.downsample_conv_1 = Downsample_Conv2d(n_filters, n_filters, bias, kernel_size, padding=1)
        self.downsample_conv_2 = Downsample_Conv2d(in_dim, n_filters, bias, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        _x = x
        _x = self.relu(_x)
        _x = self.conv(_x)
        _x = self.relu(_x)
        residual = self.downsample_conv_1(_x)
        shortcut = self.downsample_conv_2(x)

        return residual + shortcut

class ResBlock(nn.Module):
    def __init__(self, in_dim, n_filters=256):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_dim, n_filters, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        residual = x
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual
    
class Generator(nn.Module):
    def __init__(self, latent_dim: int = 128, in_dim: int=256, n_filters: int=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.linear = nn.Linear(latent_dim, 4*4*in_dim)

        self.main = nn.Sequential(
            ResnetBlockUp(in_dim=in_dim, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.linear(z)
        # reshape output of linear layer
        x = x.view(x.shape[0], -1, 4, 4)

        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, n_filters=128):
        super().__init__()
        self.main = nn.Sequential(
            ResBlockDown(3, n_filters=n_filters),
            ResBlockDown(128, n_filters=n_filters),
            ResBlock(n_filters, n_filters=n_filters),
            ResBlock(n_filters, n_filters=n_filters),
            nn.ReLU()
        )
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        
        self.linear = nn.Linear(n_filters, 1)

    def forward(self, x):
        x = self.main(x)
        x = self.global_pool(x)
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty