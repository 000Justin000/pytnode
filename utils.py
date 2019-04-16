import os
import sys
import bisect
import torch
import torch.nn as nn
import networkx as nx

# compute the running average
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
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


# This function need to be stateless
class ODEJumpFunc(nn.Module):

    def __init__(self, dim_c, dim_h, dim_N, dim_hidden=20, num_hidden=0, activation=nn.CELU(), aggregation=None, jump_type="read", evnts=[], evnt_align=False, graph=None):
        super(ODEJumpFunc, self).__init__()

        self.dim_c = dim_c
        self.dim_h = dim_h
        self.dim_N = dim_N

        self.F = GCU(dim_c, dim_h, dim_hidden, num_hidden, activation, aggregation)
        self.G = nn.Sequential(MLP(dim_c, dim_h, dim_hidden, num_hidden, activation), nn.Softplus())
        self.W = nn.ModuleList([MLP(dim_c, dim_h, dim_hidden, num_hidden, activation) for _ in range(dim_N)])
        self.L = nn.Sequential(MLP(dim_c+dim_h, dim_N, dim_hidden, num_hidden, activation), SoftPlus())

        self.set_evnts(jump_type, evnts, evnt_align)
        self.set_graph(graph)

        self.backtrace = []

    def set_evnts(self, jump_type=None, evnts=[], evnt_align=None):
        if jump_type is not None:
            assert jump_type in ["simulate", "read"], "invalide jump_type, must be one of [simulate, read]"
            self.jump_type = jump_type

        self.evnts = evnts

        if evnt_align is not None:
            self.evnt_align = evnt_align

    def set_graph(self, graph=None):
        if graph:
            self.graph = graph
        else:
            self.graph = nx.Graph()
            self.graph.add_node(0)

    def forward(self, t, z):
        assert len(z.shape) == 3, 'z need to be 3 dimensional vector accessed by [seq_id, node_id, dim_id]'

        c = z[:, :, :self.dim_c]
        h = z[:, :, self.dim_c:]

        dc = torch.stack(tuple(self.F(z[:, nid, :], z[:, list(self.graph.neighbors(nid)), :]) for nid in self.graph.nodes()), dim=1)

        # orthogonalize dc w.r.t. to c
        dc = dc - (dc*c).sum(dim=2, keepdim=True) / (c*c).sum(dim=2, keepdim=True) * c

        dh = -self.G(c) * h

        return torch.cat((dc, dh), dim=2)

    def next_simulated_jump(self, t0, z0, t1):

        if not self.evnt_align:
            m = torch.distributions.Exponential(self.L(z0).double())
            # next arrival time
            tt = t0 + m.sample()
            tt_min = tt.min()

            if tt_min <= t1:
                dN = (tt == tt_min).float()
            else:
                dN = torch.zeros(tt.shape)

            next_t = min(tt_min, t1)
        else:
            assert t0 < t1

            lmbda_dt = self.L(z0) * (t1 - t0)
            rd = torch.rand(lmbda_dt.shape)
            dN = torch.zeros(lmbda_dt.shape)
            dN[rd < lmbda_dt ** 2 / 2] += 1
            dN[rd < lmbda_dt ** 2 / 2 + lmbda_dt * torch.exp(-lmbda_dt)] += 1

            next_t = t1

        return dN, next_t

    def simulated_jump(self, dN, t, z):
        assert self.jump_type == "simulate", "simulate_jump must be called with jump_type = simulate"
        dz = torch.zeros(z.shape)
        sequence = []

        dz[:, :, self.dim_c:] += torch.matmul(torch.stack(tuple(W_(z[:, :, :self.dim_c]) for W_ in self.W), dim=-1), dN.unsqueeze(-1)).squeeze(-1)

        for idx in dN.nonzero():
            for _ in range(dN[tuple(idx)].int()):
                sequence.append((t,) + tuple(idx))
        self.evnts.extend(sequence)

        return dz

    def next_read_jump(self, t0, t1):
        assert self.jump_type == "read", "next_read_jump must be called with jump_type = read"
        assert t0 != t1, "t0 can not equal t1"

        t = t1
        inf = sys.maxsize
        if t0 < t1:  # forward
            idx = bisect.bisect_right(self.evnts, (t0, inf, inf, inf))
            if idx != len(self.evnts):
                t = min(t1, torch.tensor(self.evnts[idx][0], dtype=torch.float64))
        else:  # backward
            idx = bisect.bisect_left(self.evnts, (t0, -inf, -inf, -inf))
            if idx > 0:
                t = max(t1, torch.tensor(self.evnts[idx-1][0], dtype=torch.float64))

        assert t != t0, "t can not equal t0"
        return t

    def read_jump(self, t, z):
        assert self.jump_type == "read", "read_jump must be called with jump_type = read"
        dz = torch.zeros(z.shape)

        inf = sys.maxsize
        lid = bisect.bisect_left(self.evnts, (t, -inf, -inf, -inf))
        rid = bisect.bisect_right(self.evnts, (t, inf, inf, inf))

        dN = torch.zeros(z.shape[:-1] + (self.dim_N,))
        for evnt in self.evnts[lid:rid]:
            _, sid, nid, eid = evnt
            dN[sid, nid, eid] += 1

        dz[:, :, self.dim_c:] += torch.matmul(torch.stack(tuple(W_(z[:, :, :self.dim_c]) for W_ in self.W), dim=-1), dN.unsqueeze(-1)).squeeze(-1)

        return dz


# create the outdir
def create_outpath(dataset):
    path = os.getcwd()
    pid = os.getpid()

    wsppath = os.path.join(path, 'workspace')
    if not os.path.isdir(wsppath):
        os.mkdir(wsppath)

    outpath = os.path.join(wsppath, 'dataset:'+dataset + '-' + 'pid:'+str(pid))
    assert not os.path.isdir(outpath), 'output directory already exist (process id coincidentally the same), please retry'
    os.mkdir(outpath)

    return outpath
