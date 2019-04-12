import torch
import torch.nn as nn

# compute the running average
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.90):
        self.vals = []
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.vals = []
        self.val = None
        self.avg = 0

    def update(self, val):
        self.vals.append(val)
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


# SoftPlus activation function add epsilon
class SoftPlus(nn.Module):

    def __init__(self, beta=1.0, threshold=20, epsilon=1.0e-15):
        super(SoftPlus, self).__init__()
        self.Softplus = nn.Softplus(beta, threshold)
        self.epsilon = epsilon

    def forward(self, x):
        return self.Softplus(x) + self.epsilon


# multi-layer perceptron
class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=20, num_hidden=0, activation=nn.CELU()):
        super(MLP, self).__init__()

        if num_hidden == 0:
            self.linears = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        elif num_hidden >= 1:
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(dim_in, dim_hidden))
            self.linears.extend([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_hidden-1)])
            self.linears.append(nn.Linear(dim_hidden, dim_out))
        else:
            raise Exception('number of hidden layers must be positive')

        for m in self.linears:
            nn.init.normal_(m.weight, mean=0, std=0.1)
            nn.init.uniform_(m.bias, a=-0.1, b=0.1)

        self.activation = activation

    def forward(self, x):
        for m in self.linears[:-1]:
            x = self.activation(m(x))

        return self.linears[-1](x)


# graph convolution unit
class GCU(nn.Module):

    def __init__(self, dim_c, dim_h=0, dim_hidden=20, num_hidden=0, activation=nn.CELU(), aggregation=None):
        super(GCU, self).__init__()

        self.cur = nn.Sequential(MLP((dim_c+dim_h),   dim_hidden, dim_hidden, num_hidden, activation), activation)
        self.nbr = nn.Sequential(MLP((dim_c+dim_h)*2, dim_hidden, dim_hidden, num_hidden, activation), activation)
        self.out = nn.Linear(dim_hidden*2, dim_c)

        nn.init.normal_(self.out.weight, mean=0, std=0.1)
        nn.init.uniform_(self.out.bias, a=-0.1, b=0.1)

        if aggregation is None:
            self.aggregation = lambda vnbr: vnbr.sum(dim=-2)
        else:
            self.aggregation = aggregation

    def forward(self, z, z_):
        assert len(z.shape) >= 1,  'z  need to be >=1 dimensional vector accessed by [...         dim_id]'
        assert len(z_.shape) >= 2, 'z_ need to be >=2 dimensional vector accessed by [... nbr_id, dim_id]'

        v = self.cur(z)
        v_ = torch.zeros(v.shape) if z_.shape[-2] == 0 else self.aggregation(self.nbr(torch.cat((z.unsqueeze(-2).expand(z_.shape), z_), dim=-1)))

        dc = self.out(torch.cat((v, v_), dim=-1))

        return dc


# recurrent neural network
class RNN(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, activation):
        super(RNN, self).__init__()

        self.dim_hidden = dim_hidden
        self.i2h = MLP(dim_in+dim_hidden, dim_hidden, dim_hidden, num_hidden, activation)
        self.h2o = MLP(dim_hidden, dim_out, dim_hidden, num_hidden, activation)
        self.activation = activation

    def forward(self, x):
        assert len(x.shape) > 2,  'z need to be at least a 2 dimensional vector accessed by [tid ... dim_id]'

        hh = [torch.zeros(x.shape[1:-1] + (self.dim_hidden,))]
        for i in range(x.shape[0]):
            combined = torch.cat((x[i], hh[-1]), dim=-1)
            hh.append(self.activation(self.i2h(combined)))

        return self.h2o(torch.stack(tuple(hh[1:])))
