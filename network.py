import torch
import torch.nn as nn
from torch.nn import init
import functools

# We referred code and idea from Junyan Zhu's CycleGAN project.(https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
# Only implemented the network for style transfer mentioned in the paper. https://arxiv.org/abs/1703.10593
# -- Renjie Zhao, r2zhao at eng dot ucsd dot edu

def define_G(input_nc=3, output_nc=3, norm='instance', use_dropout=False, init_gain=0.02, device='cuda'):
    """The image of style Art for style transfer is 256x256, so the generator for style transfer is using 
    9 residual blocks: c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3

    In fact, the difference among different figure size is just the number of the R256, can be motified.

    Besides, it is using ResNet model and InstanceNorm as mentioned in the paper, we leave the ability to
    change to batch norm.

    In this code, we fixed the parameters to just get the network for style transfer.

    Parameters:
        input_nc (int)     -- the number of channels in input images (normally 3)
        output_nc (int)    -- the number of channels in output images (normally 3)
        norm (str)         -- the type of normalization layers used in the network.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        device (int list)  -- which device the network runs on. default gpu
    Returns a generator network
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = ResnetGenerator(input_nc, output_nc, norm_layer=norm_layer, use_dropout=use_dropout)
    return init_net(net, init_gain, device)


def define_D(input_nc=3, norm='instance', init_gain=0.02, device='cuda'):
    """Create a discriminator, 70x70 PatchGAN, 4x4 Convolution-InstanceNorm-LeakyReLU layer, C64-C128-C256-C512
    Parameters:
        input_nc (int)     -- the number of channels in input images (normally 3)
        norm (str)         -- the type of normalization layers used in the network. 
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        device (int list)  -- which device the network runs on. default gpu
    Returns a discriminator network
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = PatchGANDiscriminator(input_nc, norm_layer=norm_layer)
    return init_net(net, init_gain, device)


def init_net(net, init_gain=0.02, device='cuda'):
    """Initialize network weights. Only use normal initiator
    Parameters:
        net (network)        -- network to be initialized
        init_gain (float)    -- scaling factor for normal.
        device (str)         -- which device the network runs on. default gpu
    Return an initialized network.
    """

     # define the initialization function
    def init_weights(m): 
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix
        elif classname.find('BatchNorm2d') != -1: 
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)


    if not torch.cuda.is_available():
        device = 'cpu'
    net.to(device)
    net.apply(init_weights) # apply the initialization function

    return net


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class ResnetGenerator(nn.Module):
    # Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    def __init__(self, input_nc, output_nc, C=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images (normally 3)
            output_nc (int)     -- the number of channels in output images (normally 3)
            C (int)             -- the number of filters in the last conv layer (default 64)
            norm_layer          -- normalization layer (to test use instance norm or batch norm)
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        # Bias for instance norm
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # c7s1-64
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, C, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(C),
                 nn.ReLU(True)]

        # add downsampling layers: d128 (from 64 to 128), d256 (from 128 to 256)
        n_downsampling = 2
        for i in range(n_downsampling):  
            mult = 2 ** i
            model += [nn.Conv2d(C * mult, C * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(C * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        # add 9 ResNet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(C * mult, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # add upsampling layers: u128, u64. Using transposeconv since stride 1/2
        for i in range(n_downsampling):  
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(C * mult, int(C * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(C * mult / 2)),
                      nn.ReLU(True)]

        # c7s1-3
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(C, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        #Standard forward
        return self.model(input)


class ResnetBlock(nn.Module):
    # Define a Resnet block, add reflection padding to reduce artifacts

    def __init__(self, dim, norm_layer, use_dropout=False, use_bias=True):
        """Construct a ResNet block.
        Parameters:
            dim (int)          -- the number of channels in the conv layer.
            norm_layer         -- normalization layer
            use_dropout (bool)    -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        """
        super(ResnetBlock, self).__init__()

        sequence = [nn.ReflectionPad2d(1)]

        sequence += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            sequence += [nn.Dropout(0.5)]

        sequence += [nn.ReflectionPad2d(1)]

        sequence += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias), norm_layer(dim)]

        self.conv_block = nn.Sequential(*sequence)

    def forward(self, x):
        #Forward function (with skip connections)
        out = x + self.conv_block(x)
        return out


class PatchGANDiscriminator(nn.Module):
    # Define a 70x70 PatchGAN discriminator, C64-C128-C256-C512 + to 1 conv

    def __init__(self, input_nc, C=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            C (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchGANDiscriminator, self).__init__()
        # Bias for instance norm
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # do not use InstanceNorm for the first C64 layer
        sequence = [nn.Conv2d(input_nc, C, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        
        # gradually increase the number of filters, 128->256->512. InstanceNorm, bias!
        for n in range(1, n_layers+1):  
            nf_mult_prev = nf_mult
            nf_mult = 2 ** n
            sequence += [
                nn.Conv2d(C * nf_mult_prev, C * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(C * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(C * nf_mult, 1, kernel_size=4, stride=1, padding=1)]  
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # Standard forward.
        return self.model(input)