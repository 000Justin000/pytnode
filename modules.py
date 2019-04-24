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


# recurrent neural network
class RNN(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, activation):
        super(RNN, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.i2h = MLP(dim_in+dim_hidden, dim_hidden, dim_hidden, num_hidden, activation)
        self.h2o = MLP(dim_hidden, dim_out, dim_hidden, num_hidden, activation)
        self.activation = activation

    def forward(self, x, h0=None):
        assert len(x.shape) > 2,  'z need to be at least a 2 dimensional vector accessed by [tid ... dim_id]'

        if h0 is None:
            hh = [torch.zeros(x.shape[1:-1] + (self.dim_hidden,))]
        else:
            hh = [h0]

        for i in range(x.shape[0]):
            combined = torch.cat((x[i], hh[-1]), dim=-1)
            hh.append(self.activation(self.i2h(combined)))

        return self.h2o(torch.stack(tuple(hh)))


# graph convolution unit
class GCU(nn.Module):

    def __init__(self, dim_c, dim_h=0, dim_hidden=20, num_hidden=0, activation=nn.CELU(), graph=None, aggregation=None):
        super(GCU, self).__init__()

        self.cur = nn.Sequential(MLP((dim_c+dim_h),   dim_hidden, dim_hidden, num_hidden, activation), activation)
        self.nbr = nn.Sequential(MLP((dim_c+dim_h)*2, dim_hidden, dim_hidden, num_hidden, activation), activation)
        self.out = nn.Linear(dim_hidden*2, dim_c)

        nn.init.normal_(self.out.weight, mean=0, std=0.1)
        nn.init.uniform_(self.out.bias, a=-0.1, b=0.1)

        if graph is not None:
            self.graph = graph
        else:
            self.graph = nx.Graph()
            self.graph.add_node(0)

        if aggregation is None:
            self.aggregation = lambda vnbr: vnbr.sum(dim=-2)
        else:
            self.aggregation = aggregation

    def forward(self, z):
        assert len(z.shape) >= 2, 'z_ need to be >=2 dimensional vector accessed by [..., node_id, dim_id]'

        curvv = self.cur(z)

        def conv(nid):
            env = list(self.graph.neighbors(nid))
            if len(env) == 0:
                nbrv = torch.zeros(curvv[nid].shape)
            else:
                nbrv = self.aggregation(self.nbr(torch.cat((z[..., [nid]*len(env), :], z[..., env, :]), dim=-1)))
            return nbrv

        nbrvv = torch.stack([conv(nid) for nid in self.graph.nodes()], dim=-2)

        dcdt = self.out(torch.cat((curvv, nbrvv), dim=-1))

        return dcdt


# This function need to be stateless
class ODEFunc(nn.Module):

    def __init__(self, dim_c, dim_hidden=20, num_hidden=0, activation=nn.CELU(), ortho=False, graph=None, aggregation=None):
        super(ODEFunc, self).__init__()

        self.dim_c = dim_c
        self.ortho = ortho

        if graph is not None:
            self.F = GCU(dim_c, 0, dim_hidden, num_hidden, activation, aggregation, graph)
        else:
            self.F = MLP(dim_c, dim_c, dim_hidden, num_hidden, activation)

    def forward(self, t, c):
        dcdt = self.F(c)

        # orthogonalize dc w.r.t. to c
        if self.ortho:
            dcdt = dcdt - (dcdt*c).sum(dim=-1, keepdim=True) / (c*c).sum(dim=-1, keepdim=True) * c

        return dcdt


# This function need to be stateless
class ODEJumpFunc(nn.Module):

    def __init__(self, dim_c, dim_h, dim_N, dim_hidden=20, num_hidden=0, activation=nn.CELU(), ortho=False, jump_type="read", evnts=[], evnt_align=False, graph=None, aggregation=None):
        super(ODEJumpFunc, self).__init__()

        self.dim_c = dim_c
        self.dim_h = dim_h
        self.dim_N = dim_N
        self.ortho = ortho

        assert jump_type in ["simulate", "read"], "invalide jump_type, must be one of [simulate, read]"
        self.jump_type = jump_type
        assert (jump_type == "simulate" and len(evnts) == 0) or jump_type == "read"
        self.evnts = evnts
        self.evnt_align = evnt_align

        if graph is not None:
            self.F = GCU(dim_c, dim_h, dim_hidden, num_hidden, activation, aggregation, graph)
        else:
            self.F = MLP(dim_c+dim_h, dim_c, dim_hidden, num_hidden, activation)

        self.G = nn.Sequential(MLP(dim_c, dim_h, dim_hidden, num_hidden, activation), nn.Softplus())
        self.W = MLP(dim_c+dim_N, dim_h, dim_hidden, num_hidden, activation)
        self.L = nn.Sequential(MLP(dim_c+dim_h, dim_N, dim_hidden, num_hidden, activation), SoftPlus())

        self.backtrace = []

    def forward(self, t, z):
        c = z[..., :self.dim_c]
        h = z[..., self.dim_c:]

        dcdt = self.F(z)

        # orthogonalize dc w.r.t. to c
        if self.ortho:
            dcdt = dcdt - (dcdt*c).sum(dim=-1, keepdim=True) / (c*c).sum(dim=-1, keepdim=True) * c

        dhdt = -self.G(c) * h

        return torch.cat((dcdt, dhdt), dim=-1)

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

        c = z[..., :self.dim_c]
        for idx in dN.nonzero():
            # find location and type of event
            loc, k = tuple(idx[:-1]), idx[-1]
            ne = dN[tuple(idx)].int()

            # encode of event k
            kv = torch.zeros(self.dim_N)
            kv[k] = 1

            # repeat (# of events) times
            sequence.extend([(t,) + tuple(idx)] * ne)

            # add to jump
            dz[loc][self.dim_c:] += self.W(torch.cat((c[loc], kv), dim=-1)) * ne

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

        c = z[..., :self.dim_c]
        for evnt in self.evnts[lid:rid]:
            # find location and type of event
            loc, k = evnt[1:-1], evnt[-1]

            # encode of event k
            kv = torch.zeros(self.dim_N)
            kv[k] = 1

            # add to jump
            dz[loc][self.dim_c:] += self.W(torch.cat((c[loc], kv), dim=-1))

        return dz
