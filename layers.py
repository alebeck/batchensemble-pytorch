import math
import torch
from torch import nn
from torch.nn import functional as f


class BatchEnsembleConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, num_models, stride=1, padding=0, bias=True):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_models = num_models

        self.alpha = nn.Parameter(torch.empty(num_models, in_channels))
        self.gamma = nn.Parameter(torch.empty(num_models, out_channels))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        if bias:
            # use one bias vector per ensemble member
            self.bias = nn.Parameter(torch.empty(num_models, out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x):
        """
        X: Tensor of shape (B * M, C_in, H, W)
        Ensemble members should be stacked in BATCH dimension.
        Dim 0 layout:
            ------ batch elem 0, model 0 ------
            -------batch elem 1, model 0 ------
                      ...
            ------ batch elem 0, model n ------
            -------batch elem 1, model n ------
                      ...
        """
        batch_size = x.shape[0]
        examples_per_model = batch_size // self.num_models  # arguably this is the actual batch size

        alpha = self.alpha.tile(1, examples_per_model).view(batch_size, self.in_channels)[:, :, None, None]
        gamma = self.gamma.tile(1, examples_per_model).view(batch_size, self.out_channels)[:, :, None, None]

        x = self.conv(x * alpha) * gamma

        if self.bias is not None:
            bias = self.bias.tile(1, examples_per_model).view(batch_size, self.out_channels)[:, :, None, None]
            x = x + bias
        return x

    def reset_parameters(self):
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # random sign initialization for fast weights as mentioned in paper
        with torch.no_grad():
            self.alpha.bernoulli_(0.5).mul_(2).add_(-1)
            self.gamma.bernoulli_(0.5).mul_(2).add_(-1)


class BatchEnsembleConv1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, num_models, stride=1, padding=0, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_models = num_models

        self.alpha = nn.Parameter(torch.empty(num_models, in_channels))
        self.gamma = nn.Parameter(torch.empty(num_models, out_channels))
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        if bias:
            # use one bias vector per ensemble member
            self.bias = nn.Parameter(torch.empty(num_models, out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x):
        """
        X: Tensor of shape (B * M, C_in, W)
        Ensemble members should be stacked in BATCH dimension.
        Dim 0 layout:
            ------ batch elem 0, model 0 ------
            -------batch elem 1, model 0 ------
                      ...
            ------ batch elem 0, model n ------
            -------batch elem 1, model n ------
                      ...
        """
        batch_size = x.shape[0]
        examples_per_model = batch_size // self.num_models  # arguably this is the actual batch size

        alpha = self.alpha.tile(1, examples_per_model).view(batch_size, self.in_channels)[:, :, None]
        gamma = self.gamma.tile(1, examples_per_model).view(batch_size, self.out_channels)[:, :, None]

        x = self.conv(x * alpha) * gamma

        if self.bias is not None:
            bias = self.bias.tile(1, examples_per_model).view(batch_size, self.out_channels)[:, :, None]
            x = x + bias
        return x

    def reset_parameters(self):
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # random sign initialization for fast weights as mentioned in paper
        with torch.no_grad():
            self.alpha.bernoulli_(0.5).mul_(2).add_(-1)
            self.gamma.bernoulli_(0.5).mul_(2).add_(-1)


class BatchEnsembleLinear(nn.Module):

    def __init__(self, in_features, out_features, num_models, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_models = num_models

        self.W = nn.Parameter(torch.empty(out_features, in_features))
        self.r = nn.Parameter(torch.empty(num_models, in_features))
        self.s = nn.Parameter(torch.empty(num_models, out_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(num_models, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, X):
        """
        Expects input in shape (B*M, C_in), dim 0 layout:
            ------ x0, model 0 ------
            -------x0, model 1 ------
                      ...
            ------ x1, model 0 ------
            -------x1, model 1 ------
                      ...
        """
        B = X.shape[0] // self.num_models
        R = self.r.repeat(B, 1)
        S = self.s.repeat(B, 1)
        bias = self.bias.repeat(B, 1)
        # Eq. 5 from BatchEnsembles paper
        return torch.mm((X * R), self.W.T) * S + bias

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        with torch.no_grad():
            self.r.bernoulli_(0.5).mul_(2).add_(-1)
            self.s.bernoulli_(0.5).mul_(2).add_(-1)
