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

    def __init__(self, beta=1.0, threshold=20, epsilon=1.0e-15, dim=None):
        super(SoftPlus, self).__init__()
        self.Softplus = nn.Softplus(beta, threshold)
        self.epsilon = epsilon
        self.dim = dim

    def forward(self, x):
        # apply softplus to first dim dimension
        if self.dim is None:
            result = self.Softplus(x) + self.epsilon
        else:
            result = torch.cat((self.Softplus(x[..., :self.dim])+self.epsilon, x[..., self.dim:]), dim=-1)

        return result


# multi-layer perceptron
class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, dim_h=32, num_h=0, activation=nn.Sigmoid()):
        super(MLP, self).__init__()

        if num_h == 0:
            self.linears = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        elif num_h >= 1:
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(dim_in, dim_h))
            self.linears.extend([nn.Linear(dim_h, dim_h) for _ in range(num_h-1)])
            self.linears.append(nn.Linear(dim_h, dim_out))
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

    def __init__(self, dim_in, dim_out, dim_h, num_h, activation):
        super(RNN, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_h = dim_h
        self.i2h = MLP(dim_in+dim_h, dim_h, dim_h, num_h, activation)
        self.h2o = MLP(dim_h, dim_out, dim_h, num_h, activation)
        self.activation = activation

    def forward(self, x, h0=None):
        assert len(x.shape) > 2,  'z need to be at least a 2 dimensional vector accessed by [tid ... dim_id]'

        if h0 is None:
            hh = [torch.zeros(x.shape[1:-1] + (self.dim_h,))]
        else:
            hh = [h0]

        for i in range(x.shape[0]):
            combined = torch.cat((x[i], hh[-1]), dim=-1)
            hh.append(self.activation(self.i2h(combined)))

        return self.h2o(torch.stack(tuple(hh)))


# graph convolution unit
class GCU(nn.Module):

    def __init__(self, dim_z, dim_h=32, num_h=0, activation=nn.Sigmoid(), graph=None, aggregation=None):
        super(GCU, self).__init__()

        self.cur = nn.Sequential(MLP(dim_z,   dim_h, dim_h, num_h, activation), activation)  # input dimension dim_z, output dimension dim_h
        self.nbr = nn.Sequential(MLP(dim_z*2, dim_h, dim_h, num_h, activation), activation)  # input dimension dim_z*2, output dimension dim_h
        self.out = nn.Sequential(nn.Linear(dim_h*2, dim_z), nn.Tanh())  # the output is saturated with Tanh

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

        dzdt = self.out(torch.cat((curvv, nbrvv), dim=-1))

        return dzdt


# This function need to be stateless
class ODEFunc(nn.Module):

    def __init__(self, dim_z, dim_h=32, num_h=0, activation=nn.Sigmoid(), graph=None, aggregation=None):
        super(ODEFunc, self).__init__()
        self.dim_z = dim_z

        if graph is not None:
            self.F = GCU(dim_z, dim_h, num_h, activation, aggregation, graph)
        else:
            self.F = nn.Sequential(MLP(dim_z, dim_z, dim_h, num_h, activation), nn.Tanh())  # the output is saturated with Tanh

        self.G = nn.Sequential(MLP(dim_z, dim_z, dim_h, num_h, activation), nn.Softplus())  # the decay rate is set positive with SoftPlus

    def forward(self, t, z):
        dzdt = self.F(z) - self.G(z)*z  # we may consider adding a control gate to the dynamics term F

        return dzdt


class ODEJumpFunc(ODEFunc):

    def __init__(self, dim_z, dim_e, num_e, dim_h=32, num_h=0, activation=nn.Sigmoid(),
                 jump_type="read", evnts=[], evnt_embedding="discrete", evnt_align=False, graph=None, aggregation=None):
        super(ODEJumpFunc, self).__init__(dim_z, dim_h, num_h, activation, graph, aggregation)

        assert (jump_type == "simulate" and len(evnts) == 0) or jump_type == "read", "jump_type must either be simulate (empty evnts list), or read"

        self.dim_e = dim_e              # dimension for event encoding
        self.num_e = num_e              # number of different event type
        self.jump_type = jump_type
        self.evnts = evnts
        self.evnt_embedding = evnt_embedding
        self.evnt_align = evnt_align

        self.I = nn.Sequential(MLP(dim_z, dim_z, dim_h, num_h, activation), nn.Sigmoid())

        if evnt_embedding == "discrete":
            assert dim_e == num_e, "if event embedding is discrete, then use one dimension for each event type"
            self.evnt_embed = lambda k: (torch.arange(0, dim_e) == k).float()
            # output is a dim_e vector, each represent conditional intensity of a type of event
            self.L = nn.Sequential(MLP(dim_z, dim_e, dim_h, num_h, activation), SoftPlus())
        elif evnt_embedding == "continuous":
            self.evnt_embed = lambda k: torch.tensor(k)
            # output is a num_e*(1+2*dim_e) vector, represent coefficients, mean and log variance of num_e unit gaussian intensity function
            self.L = nn.Sequential(MLP(dim_z, num_e*(1+2*dim_e), dim_h, num_h, activation), SoftPlus(dim=num_e))
        else:
            raise Exception('evnt_type must either be discrete or continuous')

        # saturate the jump with Tanh (or not)
        self.W = MLP(dim_z+dim_e, dim_z, dim_h, num_h, activation)

        self.backtrace = []

    def next_simulated_jump(self, t0, z0, t1):
        lmbda = self.L(z0)[..., :self.num_e]

        if not self.evnt_align:
            m = torch.distributions.Exponential(lmbda.double())
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

            lmbda_dt = lmbda * (t1 - t0)
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

        for idx in dN.nonzero():
            # find location and type of event
            loc, k = tuple(idx[:-1]), idx[-1]
            ne = int(dN[tuple(idx)])

            for _ in range(ne):
                if self.evnt_embedding == "discrete":
                    # encode of event k
                    kv = self.evnt_embed(k)
                    sequence.extend([(t,) + loc + (k,)])
                elif self.evnt_embedding == "continuous":
                    params = self.L(z[loc])
                    gsmean = params[self.num_e*(1+self.dim_e*0):self.num_e*(1+self.dim_e*1)]
                    logvar = params[self.num_e*(1+self.dim_e*1):self.num_e*(1+self.dim_e*2)]
                    gsmean_k = gsmean[self.dim_e*k:self.dim_e*(k+1)]
                    logvar_k = logvar[self.dim_e*k:self.dim_e*(k+1)]
                    kv = self.evnt_embed(torch.randn(gsmean_k.shape) * torch.exp(0.5*logvar_k) + gsmean)
                    sequence.extend([(t,) + loc + (kv,)])

                # add to jump
                dz[loc] += self.I(z[loc]) * self.W(torch.cat((z[loc], kv), dim=-1))

        self.evnts.extend(sequence)

        return dz

    def next_read_jump(self, t0, t1):
        assert self.jump_type == "read", "next_read_jump must be called with jump_type = read"
        assert t0 != t1, "t0 can not equal t1"

        t = t1
        inf = sys.maxsize
        if t0 < t1:  # forward
            idx = bisect.bisect_right(self.evnts, (t0, inf))
            if idx != len(self.evnts):
                t = min(t1, torch.tensor(self.evnts[idx][0], dtype=torch.float64))
        else:  # backward
            idx = bisect.bisect_left(self.evnts, (t0, -inf))
            if idx != 0:
                t = max(t1, torch.tensor(self.evnts[idx-1][0], dtype=torch.float64))

        assert t != t0, "t can not equal t0"
        return t

    def read_jump(self, t, z):
        assert self.jump_type == "read", "read_jump must be called with jump_type = read"
        dz = torch.zeros(z.shape)

        inf = sys.maxsize
        lid = bisect.bisect_left(self.evnts, (t, -inf))
        rid = bisect.bisect_right(self.evnts, (t, inf))

        for evnt in self.evnts[lid:rid]:
            # find location and type of event
            loc, k = evnt[1:-1], evnt[-1]

            # encode of event k
            kv = self.evnt_embed(k)

            # add to jump
            dz[loc] += self.I(z[loc]) * self.W(torch.cat((z[loc], kv), dim=-1))

        return dz
