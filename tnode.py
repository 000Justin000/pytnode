import sys
import signal
import argparse
import numpy as np
import bisect
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torchdiffeq import odeint_adjoint as odeint

parser = argparse.ArgumentParser('tnode')
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--jump_type', type=str, default='none')
parser.add_argument('--paramr', type=str, default='params.pth')
parser.add_argument('--paramw', type=str, default='params.pth')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--nsave', type=int, default=10)
parser.add_argument('--dataset', type=str, default='exponential_hawkes')
parser.add_argument('--suffix', type=str, default='')
parser.set_defaults(restart=False, evnt_align=False)
parser.add_argument('--restart', dest='restart', action='store_true')
parser.add_argument('--evnt_align', dest='evnt_align', action='store_true')
args = parser.parse_args()


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

    def __init__(self, dim_in, dim_out, dim_hidden, num_hidden, activation):
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
            nn.init.constant_(m.bias, val=0)

        self.activation = activation

    def forward(self, x):
        for m in self.linears[:-1]:
            x = self.activation(m(x))

        return self.linears[-1](x)


# graph convolution unit
class GCU(nn.Module):

    def __init__(self, dim_z, dim_hidden, num_hidden, activation, aggregation=None):
        super(GCU, self).__init__()

        self.cur = nn.Sequential(MLP(dim_z,   dim_hidden, dim_hidden, num_hidden, activation), activation)
        self.nbr = nn.Sequential(MLP(dim_z*2, dim_hidden, dim_hidden, num_hidden, activation), activation)
        self.out = nn.Linear(dim_hidden*2, dim_z)

        nn.init.normal_(self.out.weight, mean=0, std=0.1)
        nn.init.constant_(self.out.bias, val=0)

        if aggregation is None:
            self.aggregation = lambda vnbr: vnbr.sum(dim=1)
        else:
            self.aggregation = aggregation

    def forward(self, zcur, znbr):
        assert len(zcur.shape) == 2, 'xcur need to be 2 dimensional vector accessed by [seq_id, dim_id]'
        assert len(znbr.shape) == 3, 'xnbr need to be 3 dimensional vector accessed by [seq_id, nbr_id, dim_id]'
        vcur = self.cur(zcur)
        vnbr = torch.zeros(vcur.shape) if znbr.shape[1] == 0 else self.aggregation(self.nbr(torch.cat((zcur.repeat(1, znbr.shape[1], 1), znbr), dim=2)))

        return self.out(torch.cat((vcur, vnbr), dim=1))


class ODEFunc(nn.Module):

    def __init__(self, dim_z, dim_k, dim_hidden=20, num_hidden=0, jump_type="none", evnt_record=None, graph=None, activation=nn.CELU(), aggregation=None):
        super(ODEFunc, self).__init__()

        assert jump_type in ["simulate", "read", "none"], "invalide jump_type, must be one of [simulate, read, none]"

        self.dim_z = dim_z
        self.dim_k = dim_k
        self.F = GCU(dim_z, dim_hidden, num_hidden, activation, aggregation)
        self.G = nn.Sequential(MLP(dim_z, dim_z, dim_hidden, num_hidden, activation), nn.Softplus())
        self.W = nn.ModuleList([MLP(dim_z, dim_z, dim_hidden, num_hidden, activation) for _ in range(dim_k)])
        self.L = nn.Sequential(MLP(dim_z, dim_k, dim_hidden, num_hidden, activation), SoftPlus())
        self.jump_type = jump_type
        self.evnt_record = [] if jump_type == "simulate" else evnt_record
        self.backtrace = []
        if graph:
            self.graph = graph
        else:
            self.graph = nx.Graph()
            self.graph.add_node(0)

    def forward(self, t, z):
        assert len(z.shape) == 3, 'z need to be 3 dimensional vector accessed by [seq_id, node_id, dim_id]'

        dz = torch.stack(tuple(self.F(z[:, nid, :], z[:, list(self.graph.neighbors(nid)), :]) for nid in self.graph.nodes()), dim=1)
        dz -= self.G(z) * z

        return dz

    def simulate_jump(self, t0, t1, z0, z1):
        assert t0 < t1
        dz = torch.zeros(z0.shape)
        sequence = []

        if self.jump_type == "simulate":
            lmbda_dt = (self.L(z0) + self.L(z1)) / 2 * (t1 - t0)
            rd = torch.rand(lmbda_dt.shape)
            dN = torch.zeros(lmbda_dt.shape)
            dN[rd < lmbda_dt ** 2 / 2] += 1
            dN[rd < lmbda_dt ** 2 / 2 + lmbda_dt * torch.exp(-lmbda_dt)] += 1

            dz += torch.matmul(torch.stack(tuple(W_(z1) for W_ in self.W), dim=-1), dN.unsqueeze(-1)).squeeze()
            for evnt in dN.nonzero():
                for _ in range(dN[tuple(evnt)].int()):
                    sequence.append((t1,) + tuple(evnt))
            self.evnt_record.extend(sequence)

        return dz

    def next_jump(self, t0, t1):
        assert t0 != t1, "t0 can not equal t1"

        t = t1

        if self.jump_type == "read":
            inf = sys.maxsize
            if t0 < t1:  # forward
                idx = bisect.bisect_right(self.evnt_record, (t0, inf, inf, inf))
                if idx != len(self.evnt_record):
                    t = min(t1, torch.tensor(self.evnt_record[idx][0], dtype=torch.float64))
            else:  # backward
                idx = bisect.bisect_left(self.evnt_record, (t0, -inf, -inf, -inf))
                if idx > 0:
                    t = max(t1, torch.tensor(self.evnt_record[idx-1][0], dtype=torch.float64))

        assert t != t0, "t can not equal t0"
        return t

    def read_jump(self, t1, z1):
        dz = torch.zeros(z1.shape)

        if self.jump_type == "read":
            inf = sys.maxsize
            lid = bisect.bisect_left(self.evnt_record, (t1, -inf, -inf, -inf))
            rid = bisect.bisect_right(self.evnt_record, (t1, inf, inf, inf))

            dN = torch.zeros(z1.shape[:-1] + (self.dim_k,))
            for evnt in self.evnt_record[lid:rid]:
                t, sid, nid, eid = evnt
                dN[sid, nid, eid] += 1

            dz += torch.matmul(torch.stack(tuple(W_(z1) for W_ in self.W), dim=-1), dN.unsqueeze(-1)).squeeze(-1)

        return dz


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


def read_timeseries(dat, num_vertices=1):
    timeseries = []
    for u in range(num_vertices):
        for t in dat[0][u]:
            timeseries.append((t, u, 0))

    return [sorted(timeseries), sorted(timeseries), sorted(timeseries)]


def read_timeseries(filename):
    with open(filename) as f:
        seqs = f.readlines()
    return [[(float(t), vid, 0) for vid, vts in enumerate(seq.split(";")) for t in vts.split()] for seq in seqs]


def visualize(tsave, trace, lmbda, tsave_, trace_, grid, lmbda_real, tsne, batch_id, itr, appendix=""):
    for sid in range(trace.shape[1]):
        for nid in range(trace.shape[2]):
            fig = plt.figure(figsize=(6, 6), facecolor='white')
            axe = plt.gca()
            axe.set_title('Point Process Modeling')
            axe.set_xlabel('time')
            axe.set_ylabel('intensity')
            axe.set_ylim(-10.0, 10.0)

            # plot the state function
            for dat in list(trace[:, sid, nid, :].detach().numpy().T):
                plt.plot(tsave.numpy(), dat, linewidth=0.5)

            # plot the state function (backward trace)
            if (tsave_ is not None) and (trace_ is not None):
                for dat in list(trace_[:, sid, nid, :].detach().numpy().T):
                    plt.plot(tsave_.numpy(), dat, linewidth=0.3, linestyle="dashed", color="black")

            # plot the intensity function
            plt.plot(tsave.numpy(), lmbda[:, sid, nid, :].detach().numpy(), linewidth=1.2)
            if (grid is not None) and (lmbda_real is not None):
                plt.plot(grid.numpy(), lmbda_real[sid], linewidth=0.9)

            tsne_current = [record for record in tsne if (record[1] == sid and record[2] == nid)]
            evnt_time = np.array([tsave[record[0]] for record in tsne_current])
            evnt_type = np.array([record[3] for record in tsne_current])

            plt.scatter(evnt_time, np.ones(len(evnt_time)) * 7.0, 2.0, c=evnt_type)
            plt.savefig(args.dataset + args.suffix + '/{:03d}_{:03d}_{:04d}'.format(batch_id[sid], nid, itr) + appendix, dpi=250)
            fig.clf()
            plt.close(fig)


# this function takes in a time series and create a grid for modeling it
# it takes an array of sequences of three tuples, and extend it to four tuple
def create_tsave(tmin, tmax, dt, batch_record, evnt_align=False):
    """
    :param tmin: min time of sequence
    :param tmax: max time of the sequence
    :param dt: step size
    :param batch_record: 4-tuple (raw_time, sid, nid, eid)
    :param evnt_align: whether to round the event time up to the next grid point
    :return tsave: the time to save state in ODE simulation
    :return gtid: grid time id
    :return evnt_record: 4-tuple (rounded_time, sid, nid, eid)
    :return tsne: 4-tuple (event_time_id, sid, nid, eid)
    """

    if evnt_align:
        tc = lambda t: np.round(np.ceil((t-tmin) / dt) * dt + tmin, decimals=8)
    else:
        tc = lambda t: t

    evnt_record = [(tc(record[0]),) + record[1:] for record in batch_record if tmin < tc(record[0]) < tmax]

    grid = np.round(np.arange(tmin, tmax+dt, dt), decimals=8)
    evnt = np.array([record[0] for record in evnt_record])
    tsave = np.sort(np.unique(np.concatenate((grid, evnt))))
    t2tid = {t: tid for tid, t in enumerate(tsave)}

    # g(rid)tid
    # t(ime)s(equence)n(ode)e(vent)
    gtid = [t2tid[t] for t in grid]
    tsne = [(t2tid[record[0]],) + record[1:] for record in evnt_record]

    return torch.tensor(tsave), gtid, evnt_record, tsne


def forward_pass(func, z0, tspan, dt, batch):
    # merge the sequences to create a sequence
    batch_record = sorted([(record[0],) + (sid,) + record[1:]
                           for sid in range(len(batch)) for record in batch[sid]])

    # set up grid
    tsave, gtid, evnt_record, tsne = create_tsave(tspan[0], tspan[1], dt, batch_record, args.evnt_align)
    func.evnt_record = evnt_record

    # forward pass
    trace = odeint(func, z0.repeat(len(batch), 1, 1), tsave, method='jump_adams', rtol=1.0e-7, atol=1.0e-9)
    lmbda = func.L(trace)
    loss = -(sum([torch.log(lmbda[record]) for record in tsne]) - (lmbda[gtid, :, :, :] * dt).sum())

    return tsave, trace, lmbda, gtid, tsne, loss


def exponential_hawkes_lmbda(tmin, tmax, dt, lmbda0, alpha, beta, TS, evnt_align=False):
    if evnt_align:
        tc = lambda t: np.round(np.ceil((t-tmin) / dt) * dt + tmin, decimals=8)
    else:
        tc = lambda t: t

    cl = lambda t: np.round(np.ceil((t-tmin) / dt) * dt + tmin, decimals=8)
    grid = np.round(np.arange(tmin, tmax+dt, dt), decimals=8)
    t2tid = {t: tid for tid, t in enumerate(grid)}

    lmbda = []
    kernel = alpha * np.exp(-beta * np.arange(0.0, 10.0/beta, dt))

    for ts in TS:
        vv = np.zeros(grid.shape)
        for record in ts:
            vv[t2tid[cl(record[0])]] = np.exp(-beta * (cl(record[0]) - tc(record[0])))
        lmbda.append(lmbda0 + np.convolve(kernel, vv)[:grid.shape[0]])

    return lmbda


def powerlaw_hawkes_lmbda(tmin, tmax, dt, lmbda0, alpha, beta, sigma, TS, evnt_align=False):
    cl = lambda t: np.round(np.ceil((t-tmin) / dt) * dt + tmin, decimals=8)
    grid = np.round(np.arange(tmin, tmax+dt, dt), decimals=8)
    t2tid = {t: tid for tid, t in enumerate(grid)}

    lmbda = []
    kernel_grid = np.arange(dt, 10.0 + 10.0*sigma, dt)
    kernel = np.concatenate(([0], alpha * (beta/sigma) * (kernel_grid > sigma) * (kernel_grid / sigma)**(-beta-1)))

    if evnt_align:
        for ts in TS:
            vv = np.zeros(grid.shape)
            for record in ts:
                vv[t2tid[cl(record[0])]] = 1.0
            lmbda.append(lmbda0 + np.convolve(kernel, vv)[:grid.shape[0]])
    else:
        raise Exception("option not implemented")

    return lmbda


def self_inhibiting_lmbda(tmin, tmax, dt, mu, beta, TS):
    grid = np.round(np.arange(tmin, tmax+dt, dt), decimals=8)

    lmbda = []
    for ts in TS:
        lmbda0 = mu * (grid-tmin)
        for record in ts:
            lmbda0[grid > record[0]] -= beta
        lmbda.append(np.exp(lmbda0))

    return lmbda


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

    # create a graph
    G = nx.Graph()
    G.add_node(0)

    dim_z, dim_k, dt, tspan = 5, 1, 0.05, (0.0, 100.0)
    path = "literature_review/MultiVariatePointProcess/experiments/data/"
    TSTR = read_timeseries(path + args.dataset + "_training.csv")
    TSVA = read_timeseries(path + args.dataset + "_validation.csv")
    TSTE = read_timeseries(path + args.dataset + "_testing.csv")

    if args.dataset == "exponential_hawkes":
        lmbda_va_real = exponential_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 1.0, TSVA, args.evnt_align)
    elif args.dataset == "powerlaw_hawkes":
        lmbda_va_real = powerlaw_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 2.0, 1.0, TSVA, args.evnt_align)
    elif args.dataset == "self_inhibiting":
        lmbda_va_real = self_inhibiting_lmbda(tspan[0], tspan[1], dt, 0.5, 0.2, TSVA)

    # initialize / load model
    torch.manual_seed(0)
    func = ODEFunc(dim_z, dim_k, dim_hidden=20, num_hidden=0, jump_type=args.jump_type, graph=G, activation=nn.CELU())
    if args.restart:
        checkpoint = torch.load(args.dataset + args.suffix + "/" + args.paramr)
        func.load_state_dict(checkpoint['func_state_dict'])
        z0 = checkpoint['z0']
        it0 = checkpoint['it0']
    else:
        z0 = torch.zeros(G.number_of_nodes(), dim_z, requires_grad=True)
        it0 = 0

    optimizer = optim.Adam([{'params': func.parameters()},
                            {'params': z0, 'lr': 1e-2},
                            ], lr=1e-4, weight_decay=1e-6)

    loss_meter = RunningAverageMeter()

    # if read from history, then fit to maximize likelihood
    it = it0
    if func.jump_type == "read":
        while it < args.niters:
            # clear out gradients for variables
            optimizer.zero_grad()

            # sample a mini-batch, create a grid based on that
            np.random.seed(it)
            batch_id = np.random.choice(len(TSTR), args.batch_size, replace=False)
            batch = [TSTR[seqid] for seqid in batch_id]

            # forward pass
            tsave, trace, lmbda, gtid, tsne, loss = forward_pass(func, z0, tspan, dt, batch)
            loss_meter.update(loss.item() / len(batch))
            print("iter: {}, running ave loss: {:.4f}".format(it, loss_meter.avg), flush=True)

            # backward prop
            func.backtrace.clear()
            loss.backward()

            # step
            optimizer.step()

            it = it+1

            # save
            torch.save({'func_state_dict': func.state_dict(), 'z0': z0, 'it0': it}, args.dataset + args.suffix + '/' + args.paramw)

            # validate and visualize
            if it % args.nsave == 0:
                # use the full validation set for forward pass
                tsave, trace, lmbda, gtid, tsne, loss = forward_pass(func, z0, tspan, dt, TSVA)
                print("iter: {}, validation loss: {:.4f}".format(it, loss.item()/len(TSVA)), flush=True)

                # backward prop
                func.backtrace.clear()
                loss.backward()

                # visualize
                tsave_ = torch.tensor([record[0] for record in reversed(func.backtrace)])
                trace_ = torch.stack(tuple(record[1] for record in reversed(func.backtrace)))
                visualize(tsave, trace, lmbda, tsave_, trace_, tsave[gtid], lmbda_va_real, tsne, range(len(TSVA)), it)


    # simulate for validation set
    func.jump_type = "simulate"
    tsave, trace, lmbda, gitd, tsne, loss = forward_pass(func, z0, tspan, dt, [[]]*len(TSVA))
    visualize(tsave, trace, lmbda, None, None, None, None, tsne, range(len(TSVA)), it, "simulate")

    # computing testing error
    # func.jump_type = "read"
    # tsave, trace, lmbda, gtid, tsne, loss = forward_pass(func, z0, tspan, dt, TSTE)
    # print("iter: {}, testing loss: {:.4f}".format(it, loss.item()/len(TSTE)))
