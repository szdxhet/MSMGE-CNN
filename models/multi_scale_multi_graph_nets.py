import numpy as np
import torch as th
from torch import nn
from braindecode.torch_ext.init import glorot_weight_zero_bias
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var


from abc import abstractmethod
from .utils import normalize_adj


from torch import nn
import torch
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1., **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:

            self.weight.data = th.renorm(self.weight.data, p=2, dim=0,
                                         maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1., **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = th.renorm(self.weight.data, p=2, dim=0,
                                         maxnorm=self.max_norm)
        return super(LinearWithConstraint, self).forward(x)


def _transpose_to_0312(x):
    return x.permute(0, 3, 1, 2)


def _transpose_to_0132(x):
    return x.permute(0, 1, 3, 2)


def _review(x):
    return x.contiguous().view(-1, x.size(2), x.size(3))


def _squeeze_final_output(x):
    """
    Remove empty dim at end and potentially remove empty time dim
    Do not just use squeeze as we never want to remove first dim
    :param x:
    :return:
    """
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


class GraphEmbedding(nn.Module):
    def __init__(self,
                 n_nodes,
                 input_dim,
                 adj1,
                 adj2,
                 k=1,
                 adj_learn=True):
        super(GraphEmbedding, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self.xs, self.ys = th.tril_indices(self.n_nodes, self.n_nodes, offset=-1)
        node_value1 = adj1[self.xs, self.ys]
        node_value2 = adj2[self.xs, self.ys]
        self.edge_weight1 = nn.Parameter(node_value1.clone().detach(), requires_grad=self.adj_learn)
        self.edge_weight2 = nn.Parameter(node_value2.clone().detach(), requires_grad=self.adj_learn)

    def forward(self, x):
        edge_weight1 = th.zeros([self.n_nodes, self.n_nodes], device=x.device)
        edge_weight2 = th.zeros([self.n_nodes, self.n_nodes], device=x.device)

        xx = self.xs.to(x.device)
        yy = self.ys.to(x.device)
        ee1 = self.edge_weight1.to(x.device)
        ee2 = self.edge_weight2.to(x.device)

        edge_weight1[xx, yy] = ee1
        edge_weight2[xx, yy] = ee2

        edge_weight1 = edge_weight1 + edge_weight1.T + th.eye(self.n_nodes, dtype=edge_weight1.dtype, device=x.device)
        edge_weight2 = edge_weight2 + edge_weight2.T + th.eye(self.n_nodes, dtype=edge_weight2.dtype, device=x.device)
        edge_weight1 = normalize_adj(edge_weight1, mode='row')
        edge_weight2 = normalize_adj(edge_weight2, mode='row')
        x_out = [x]

        edge_weight1_iter = edge_weight1
        edge_weight2_iter = edge_weight2

        for k in range(self.k):

            x1 = th.matmul(edge_weight1_iter.unsqueeze(0), x)
            x2 = th.matmul(edge_weight2_iter.unsqueeze(0), x)
            x_out.append(x1)
            x_out.append(x2)


            edge_weight1_iter = th.matmul(edge_weight1_iter, edge_weight1)
            edge_weight2_iter = th.matmul(edge_weight2_iter, edge_weight2)

        x_out = th.stack(x_out)
        return x_out


class MSMGECNN(BaseModel):
    def __init__(self,
                 Adj1,
                 Adj2,
                 in_chans,
                 n_classes,
                 k=2,
                 input_time_length=None,
                 Adj_learn=False,
                 drop_prob=0.25,
                 pool_mode='mean',
                 f1=8,
                 f2=16,
                 kernel_length=64,
                 third_kernel_size=(8, 4),
                 final_conv_length='auto',
                 d=2,
                 ):
        super(MSMGECNN, self).__init__()

        if final_conv_length == 'auto':
            assert input_time_length is not None

        self.__dict__.update(locals())
        del self.self



        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        self.temporal_conv1 = nn.Sequential(
            Expression(_transpose_to_0312),
            Conv2dWithConstraint(in_channels=1, out_channels=self.f1,
                                 kernel_size=(1, int(self.kernel_length/2)),
                                 max_norm=None,
                                 stride=1,
                                 bias=False,
                                 padding=(0, self.kernel_length // 2)
                                 ),
            nn.BatchNorm2d(self.f1, momentum=0.01, affine=True, eps=1e-3),
        )

        self.temporal_conv2 = nn.Sequential(
            Expression(_transpose_to_0312),
            Conv2dWithConstraint(in_channels=1, out_channels=self.f1,
                                 kernel_size=(1, self.kernel_length),
                                 max_norm=None,
                                 stride=1,
                                 bias=False,
                                 padding=(0, self.kernel_length // 2)
                                 ),
            nn.BatchNorm2d(self.f1, momentum=0.01, affine=True, eps=1e-3),
        )

        self.temporal_conv3 = nn.Sequential(
            Expression(_transpose_to_0312),
            Conv2dWithConstraint(in_channels=1, out_channels=self.f1,
                                 kernel_size=(1, int(self.kernel_length*2)),
                                 max_norm=None,
                                 stride=1,
                                 bias=False,
                                 padding=(0, self.kernel_length // 2)
                                 ),
            nn.BatchNorm2d(self.f1, momentum=0.01, affine=True, eps=1e-3),
        )

        self.ge = nn.Sequential(
            Expression(_review),
            GraphEmbedding(self.in_chans, self.input_time_length, adj1=self.Adj1,adj2=self.Adj2, k=self.k, adj_learn=self.Adj_learn),
        )


        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(self.f1 * (2*self.k + 1), self.f1 * self.d * (2*self.k + 1), (self.in_chans, 1),
                                 max_norm=1, stride=1, bias=False,
                                 groups=self.f1 * (2*self.k + 1), padding=(0, 0)),
            nn.BatchNorm2d(self.f1 * self.d * (2*self.k + 1), momentum=0.01, affine=True,
                           eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        )

        self.separable_conv = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            Conv2dWithConstraint(self.f1 * self.d * (2*self.k + 1), self.f1 * self.d * (2*self.k + 1), (1, 16),
                                 max_norm=None,
                                 stride=1,
                                 bias=False, groups=self.f1 * self.d * (2*self.k + 1),
                                 padding=(0, 8)),
            Conv2dWithConstraint(self.f1 * self.d * (2*self.k + 1), self.f2, (1, 1), max_norm=None, stride=1, bias=False,
                                 padding=(0, 0)),
            nn.BatchNorm2d(self.f2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            pool_class(kernel_size=(1, 8), stride=(1, 8))
        )

        out = np_to_var(np.ones((1, self.in_chans, self.input_time_length, 1), dtype=np.float32))
        out = self.forward_init(out)
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        if self.final_conv_length == 'auto':
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time


        self.cls = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            Conv2dWithConstraint(self.f2, self.n_classes,
                                 (n_out_virtual_chans, self.final_conv_length), max_norm=0.25,
                                 bias=True),
            Expression(_transpose_to_0132),
            Expression(_squeeze_final_output)
        )

        self.apply(glorot_weight_zero_bias)



    def forward_init(self, x):
        with th.no_grad():
            batch_size = x.size(0)

            x1 = self.temporal_conv1(x)
            x1 = self.ge(x1)
            x1 = x1.view((2*self.k + 1), batch_size, -1, x1.size(-2), x1.size(-1))
            x1 = x1.permute(1, 0, 2, 3, 4).contiguous().view(batch_size, -1, x1.size(-2),x1.size(-1))
            x1 = self.spatial_conv(x1)
            x1 = self.separable_conv(x1)

            x2 = self.temporal_conv2(x)
            x2 = self.ge(x2)
            x2 = x2.view((2*self.k + 1), batch_size, -1, x2.size(-2), x2.size(-1))
            x2 = x2.permute(1, 0, 2, 3, 4).contiguous().view(batch_size, -1, x2.size(-2),x2.size(-1))
            x2 = self.spatial_conv(x2)
            x2 = self.separable_conv(x2)

            x3 = self.temporal_conv3(x)
            x3 = self.ge(x3)
            x3 = x3.view((2*self.k + 1), batch_size, -1, x3.size(-2), x3.size(-1))
            x3 = x3.permute(1, 0, 2, 3, 4).contiguous().view(batch_size, -1, x3.size(-2),x3.size(-1))
            x3 = self.spatial_conv(x3)
            x3 = self.separable_conv(x3)

            x = torch.cat((x1, x2, x3), 3)
        return x

    def forward(self, x):


        batch_size = x.size(0)
        x = x[:, :, :, None]

        x1 = self.temporal_conv1(x)
        x1 = self.ge(x1)
        x1 = x1.view((2*self.k + 1), batch_size, -1, x1.size(-2), x1.size(-1))
        x1 = x1.permute(1, 0, 2, 3, 4).contiguous().view(batch_size, -1, x1.size(-2), x1.size(-1))
        x1 = self.spatial_conv(x1)
        x1 = self.separable_conv(x1)

        x2 = self.temporal_conv2(x)
        x2 = self.ge(x2)
        x2 = x2.view((2*self.k + 1), batch_size, -1, x2.size(-2), x2.size(-1))
        x2 = x2.permute(1, 0, 2, 3, 4).contiguous().view(batch_size, -1, x2.size(-2), x2.size(-1))
        x2 = self.spatial_conv(x2)
        x2 = self.separable_conv(x2)

        x3 = self.temporal_conv3(x)
        x3 = self.ge(x3)
        x3 = x3.view((2*self.k + 1), batch_size, -1, x3.size(-2), x3.size(-1))
        x3 = x3.permute(1, 0, 2, 3, 4).contiguous().view(batch_size, -1, x3.size(-2), x3.size(-1))
        x3 = self.spatial_conv(x3)
        x3 = self.separable_conv(x3)

        x_cat = torch.cat((x1, x2, x3), 3)
        x = self.cls(x_cat)

        return x

