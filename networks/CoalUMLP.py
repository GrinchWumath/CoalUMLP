# --------------------------------------------------------
# CoalUMLP
# Licensed under The MIT License [see LICENSE for details]
# Written by Ruoyu Wu,China University Of Mining Technology,Beijing
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



class MyNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyNorm, self).__init__()
        self.group_norm = nn.GroupNorm(1, num_channels)

    def forward(self, x):
        x_shape = x.shape
        x = x.view(x_shape[0], x_shape[1], -1)  # Flatten the spatial dimensions
        x = self.group_norm(x)
        x = x.view(x_shape)  # Reshape the spatial dimensions back to their original size
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AxialShift(nn.Module):
    def __init__(self, dim, shift_sizes, dilation_rate,dims=('D', 'H', 'W'), as_bias=True, drop=0., proj_drop=0.,
                 use_multi_scale=False, use_masked=False, mask_prob=0.15):
        super().__init__()
        self.dim = dim
        self.dims = dims
        self.shift_sizes = shift_sizes
        self.pad = [shift_size // 2 for shift_size in shift_sizes]
        self.use_multi_scale = use_multi_scale
        self.use_masked = use_masked
        self.mask_prob = mask_prob
        self.dilation_rate = dilation_rate

        self.conv1 = nn.ModuleList([nn.Conv3d(dim, dim, 1, 1, 0, groups=1, bias=as_bias, dilation=dilation_rate) for _ in shift_sizes])
        self.conv2_1 = nn.ModuleList([nn.Conv3d(dim, dim, 1, 1, 0, groups=1, bias=as_bias, dilation=dilation_rate) for _ in shift_sizes])
        self.conv2_2 = nn.ModuleList([nn.Conv3d(dim, dim, 1, 1, 0, groups=1, bias=as_bias, dilation=dilation_rate) for _ in shift_sizes])
        self.conv2_3 = nn.ModuleList([nn.Conv3d(dim, dim, 1, 1, 0, groups=1, bias=as_bias, dilation=dilation_rate) for _ in shift_sizes])
        self.conv3 = nn.ModuleList([nn.Conv3d(dim, dim, 1, 1, 0, groups=1, bias=as_bias, dilation=dilation_rate) for _ in shift_sizes])
        self.actn = nn.GELU()
        self.drop = nn.Dropout(drop)

        self.norm1 = MyNorm(dim)
        self.norm2 = MyNorm(dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_orig = x.clone()
        x_masked=torch.zeros_like(x)


        if self.use_masked:
            mask = torch.rand_like(x) < self.mask_prob
            mask = mask.float()
            x = x * (1 - mask)
            x_masked=self.axial_shift_single_scale(x,0)

        if self.use_multi_scale:
            outputs = []
            for idx in range(len(self.shift_sizes)):
                output = self.axial_shift_single_scale(x, idx)
                outputs.append(output)
            x = torch.sum(torch.stack(outputs), dim=0)
        else:
            x = self.axial_shift_single_scale(x, 0)

        # Residual connection
        x = x + x_orig+x_masked
        return x

    def axial_shift_single_scale(self, x,idx):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, C, D, H, W = x.shape
        self.input_shape = (B, C, D, H, W)
        x = self.conv1[idx](x)
        x = self.norm1(x)
        x = self.actn(x)

        x_d, x_h, x_w = 0, 0, 0

        if 'D' in self.dims:
            x = F.pad(x, (3, 3, 3, 3, 3, 3), "constant", 0)
            xs = torch.chunk(x, self.shift_sizes[idx], 1)
            x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad[idx], self.pad[idx] + 1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad[idx], D)
            x_cat = torch.narrow(x_cat, 3, self.pad[idx], H)
            x_s = torch.narrow(x_cat, 4, self.pad[idx], W)
            x_shift_d = self.conv2_1[idx](x_s)
            x_d = self.actn(x_shift_d)
        if 'H' in self.dims:
            x = F.pad(x, (3, 3, 3, 3, 3, 3), "constant", 0)
            xs = torch.chunk(x, self.shift_sizes[idx], 1)
            x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad[idx], self.pad[idx] + 1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad[idx], D)
            x_cat = torch.narrow(x_cat, 3, self.pad[idx], H)
            x_s = torch.narrow(x_cat, 4, self.pad[idx], W)

            x_shift_h = self.conv2_2[idx](x_s)
            x_h = self.actn(x_shift_h)

        if 'W' in self.dims:
            x = F.pad(x, (3, 3, 3, 3, 3, 3), "constant", 0)
            xs = torch.chunk(x, self.shift_sizes[idx], 1)
            x_shift = [torch.roll(x_c, shift, 4) for x_c, shift in zip(xs, range(-self.pad[idx], self.pad[idx] + 1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad[idx], D)
            x_cat = torch.narrow(x_cat, 3, self.pad[idx], H)
            x_s = torch.narrow(x_cat, 4, self.pad[idx], W)

            x_shift_w = self.conv2_3[idx](x_s)
            x_w = self.actn(x_shift_w)

        x = x_d + x_h + x_w
        x = self.norm2(x)

        x = self.conv3[idx](x)

        return x


class AxialShiftedBlock(nn.Module):
    def __init__(self, dim, input_resolution, dilation_rate,shift_sizes,
                 mlp_ratio=4., as_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, use_multi_scale=False, use_masked=False, mask_prob=0.15):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.shift_sizes = shift_sizes
        self.mlp_ratio = mlp_ratio

        self.norm1 = MyNorm(dim)
        self.axial_shift = AxialShift(dim, shift_sizes,dilation_rate, as_bias=as_bias, drop=drop, proj_drop=drop,
                                      use_multi_scale=use_multi_scale, use_masked=use_masked, mask_prob=mask_prob)

        self.drop_path = DropPath(drop_path) if (drop_path is not None and drop_path > 0.) else nn.Identity()

        self.norm2 = MyNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        shortcut = x
        x = self.norm1(x)

        # Axial Shift block
        x = self.axial_shift(x)  # B, C, D, H, W

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x




class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Conv3d(8 * dim, 2 * dim, 1, 1, bias=False)
        self.norm = MyNorm(8 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, C, D, H, W = x.shape
        # assert L == H * W, "input feature has wrong size"
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, f"x size ({D}*{H}*{W}) are not even."

        x = x.view(B, C, D, H, W)

        # B C D/2 H/2 W/2
        x0 = x[:, :, 0::2, 0::2, 0::2]  # B C D/2 H/2 W/2
        x1 = x[:, :, 1::2, 0::2, 0::2]  # B C D/2 H/2 W/2
        x2 = x[:, :, 0::2, 1::2, 0::2]  # B C D/2 H/2 W/2
        x3 = x[:, :, 1::2, 1::2, 0::2]  # B C D/2 H/2 W/2
        x4 = x[:, :, 0::2, 0::2, 1::2]  # B C D/2 H/2 W/2
        x5 = x[:, :, 1::2, 0::2, 1::2]  # B C D/2 H/2 W/2
        x6 = x[:, :, 0::2, 1::2, 1::2]  # B C D/2 H/2 W/2
        x7 = x[:, :, 1::2, 1::2, 1::2]  # B C D/2 H/2 W/2
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], 1)  # B 8*C D/2 H/2 W/2

        x = self.norm(x)
        x = self.reduction(x)

        return x



class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(128, 128, 128), patch_size=(4, 4, 4), in_chans=4, embed_dim=64):
        super().__init__()
        # img_size = to_3tuple(img_size)
        # patch_size = to_3tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.norm = MyNorm(embed_dim)

    def forward(self, x):

        x = self.proj(x)  # x的形状为(B, embed_dim, Pd, Ph, Pw),Pd、Ph和Pw分别是三个维度上的patch数量。
        _, _, D, H, W = x.shape
        x = self.norm(x)
        return x, D, H, W


class Upsample(nn.Module):
    def __init__(self, dim,scale_factor=2):
        super(Upsample, self).__init__()
        self.dim=dim
        self.scale_factor = scale_factor
        self.conv=nn.Conv3d(dim,dim//2,kernel_size=1)
        self.norm = MyNorm(dim)

    def forward(self, x):
        B, C, D, H, W = x.shape  # Get the current depth, height, and width
        D_up = int(D * self.scale_factor)
        H_up = int(H * self.scale_factor)
        W_up = int(W * self.scale_factor)

        x_upsampled = F.interpolate(x, size=(D_up, H_up, W_up), mode='trilinear', align_corners=False)
        x_upsampled = self.norm(x_upsampled)
        x_upsampled= self.conv(x_upsampled)
        return x_upsampled

class CoalUMLPBlock(nn.Module):
    def __init__(self, dim, input_resolution, depth, shift_sizes,
                 mlp_ratio=4., as_bias=True, drop=0.,
                 drop_path=0., downsample=None,upsample=None,use_multi_scale=False, use_masked=False, mask_prob=0.15,
                 use_checkpoint=False, layer_depth=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.shift_sizes=shift_sizes

        # build blocks
        self.blocks = nn.ModuleList([
            AxialShiftedBlock(dim=dim, input_resolution=input_resolution,
                              shift_sizes=shift_sizes,
                              mlp_ratio=mlp_ratio,
                              as_bias=as_bias,
                              drop=drop,
                              dilation_rate=3 if layer_depth == 0 else 2 if layer_depth == 1 else 1,
                              drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                              use_multi_scale=use_multi_scale,
                              use_masked=use_masked,
                              mask_prob=mask_prob)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

        if upsample is not None:
            self.upsample = upsample(dim=dim)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        if self.upsample is not None:
            x=self.upsample(x)
        return x




class CoalUMLP(nn.Module):
    def __init__(self, img_size=(128, 128, 128), patch_size=(4, 4, 4), in_chans=1, num_classes=2,
                 embed_dim=64, depth=[3, 3, 3],
                 mlp_ratio=4., as_bias=True, drop_rate=0.1, mask_prob=0.15,
                 use_checkpoint=False,shift_sizes=[3,5,7,9]):

        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depth)
        self.embed_dim = embed_dim

        # Input embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        D, H, W = self.patch_embed.patches_resolution  # 代表三个维度上被分成的patch数量

        # encoder
        dpr = [x.item() for x in torch.linspace(0, drop_rate, sum(depth))]  # stochastic depth decay rule
        cur = 0
        self.layers = nn.ModuleList()
        layer_shift_sizes = shift_sizes.copy()
        for k, num_blocks in enumerate(depth):
            input_resolution = (D, H, W)
            layer_shift_sizes = [max(3, s // (2 ** k)) for s in shift_sizes]
            layer = CoalUMLPBlock(dim=self.embed_dim, input_resolution=input_resolution,
                                  depth=num_blocks, shift_sizes=layer_shift_sizes,
                                  mlp_ratio=mlp_ratio, as_bias=as_bias, drop=drop_rate,
                                  drop_path=dpr[cur: cur + num_blocks], downsample=PatchMerging if (k < self.num_layers-1) else None,
                                  upsample=None,
                                  use_multi_scale=True,
                                  use_masked=False,
                                  mask_prob=mask_prob,
                                  use_checkpoint=use_checkpoint,
                                 layer_depth=k)
            if k < self.num_layers-1:
                D, H, W = D // 2, H // 2, W // 2
                self.embed_dim = self.embed_dim * 2

            self.layers.append(layer)
            cur += num_blocks
        '''
        # Middle layer
        self.middle_layer = nn.Sequential(
            nn.Conv3d(self.embed_dim, self.embed_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.embed_dim * 2, self.embed_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
            )
            '''

        # decoder
        self.norm = MyNorm(self.embed_dim)
        self.up_layers = nn.ModuleList()
        for k, num_blocks in reversed(list(enumerate(depth))):  # Reverse the loop order
            layer_shift_sizes = [max(3, s // (2 ** k)) for s in
                                 shift_sizes]
            layer = CoalUMLPBlock(dim=self.embed_dim, input_resolution=(D, H, W),
                                  depth=num_blocks, shift_sizes=layer_shift_sizes, mlp_ratio=mlp_ratio,
                                  as_bias=as_bias, drop=drop_rate, drop_path=None, downsample=None,
                                  upsample=Upsample if (k >0) else None,
                                  use_multi_scale=False,
                                  use_masked=True,
                                  mask_prob=mask_prob,
                                  use_checkpoint=use_checkpoint,layer_depth=k)
            self.up_layers.append(layer)
            if k > 0:  # Add this condition to avoid updating D, H, W, and embed_dim for the last layer
                self.embed_dim = self.embed_dim // 2
                D, H, W = D * 2, H * 2, W * 2  # Update D, H, and W in the reverse order

        self.up_layers_without_mask= nn.ModuleList()
        embed_dim_copy = self.embed_dim * (2 ** (self.num_layers - 2))
        for k, num_blocks in reversed(list(enumerate(depth))):  # Reverse the loop order
            layer_shift_sizes = [max(3, s // (2 ** k)) for s in
                                    shift_sizes]
            layer = CoalUMLPBlock(dim=embed_dim_copy, input_resolution=(D, H, W),
                                    depth=num_blocks, shift_sizes=layer_shift_sizes, mlp_ratio=mlp_ratio,
                                    as_bias=as_bias, drop=drop_rate, drop_path=None, downsample=None,
                                    upsample=None,
                                    use_multi_scale=False,
                                    use_masked=False,
                                    mask_prob=mask_prob,
                                    use_checkpoint=use_checkpoint,layer_depth=k)

            self.up_layers_without_mask.append(layer)
            if k > 0:  # Add this condition to avoid updating D, H, W, and embed_dim for the last layer
                embed_dim_copy = embed_dim_copy // 2

        self.proj = nn.Conv3d(embed_dim, num_classes, kernel_size=1)
        self.deproj = nn.ConvTranspose3d(num_classes, num_classes, kernel_size=self.patch_embed.patch_size,
                                         stride=self.patch_embed.patch_size)

    def forward(self, x):
        x, D, H, W = self.patch_embed(x)
        layer_outputs = []
        layer_outputs.append(x)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            layer_outputs.append(x)

        x = self.norm(x)
        #x = self.middle_layer(x)
        for up_layer, up_layer_without_masks, skip_output in zip(self.up_layers,self.up_layers_without_mask,
                                                                 layer_outputs[-3::-1]):
            D, H, W = skip_output.shape[2:]

            x = up_layer(x)

            x += skip_output

            x = up_layer_without_masks(x)

        x = self.proj(x)
        x = self.deproj(x)  # Add this line to upsample back to the original image size
        x = x.view(x.shape[0], self.num_classes, self.patch_embed.img_size[0], self.patch_embed.img_size[1],
                   self.patch_embed.img_size[2])

        return x
