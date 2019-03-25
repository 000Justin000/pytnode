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
parser.add_argument('--dataset', type=str, default="exponential_hawkes")
parser.set_defaults(restart=False, evnt_align=False)
parser.add_argument('--restart', dest='restart', action='store_true')
parser.add_argument('--evnt_align', dest='evnt_align', action='store_true')
args = parser.parse_args()


# exponential activation function
class Exponential(nn.Module):
    def __init__(self, beta=1.0):
        super(Exponential, self).__init__()
        self.beta = beta

    def forward(self, x):
        return torch.exp(self.beta * x)


class ODEFunc(nn.Module):

    def __init__(self, p, q, nhidden=20, jump_type="simulate", evnt_record=None, graph=None, aggregate_func=None, decay_rate=1.0e-3):
        super(ODEFunc, self).__init__()

        assert jump_type in ["simulate", "read", "none"], "invalide jump_type, must be one of [simulate, read, none]"
        self.p = p
        self.q = q
        self.jump_type = jump_type
        self.Fc = nn.Sequential(nn.Linear(p+q, p), nn.Softplus())
        self.Fh = nn.Sequential(nn.Linear(p+q, q), nn.Softplus())
        self.Gc = nn.Sequential(nn.Linear(p+q, nhidden), nn.CELU(), nn.Linear(nhidden, nhidden), nn.CELU(), nn.Linear(nhidden, p))
        self.Gh = nn.Sequential(nn.Linear(p+q, q), nn.Softplus())
        self.L = nn.Sequential(nn.Linear(p+q, q), Exponential())
        self.A = nn.Sequential(nn.Linear(2*q, q), nn.Softplus())
        if graph:
            self.graph = graph
        else:
            self.graph = nx.Graph()
            self.graph.add_node(0)
        if aggregate_func:
            self.aggregate_func = aggregate_func
        else:
            self.aggregate_func = lambda vnb: torch.zeros(vnb.shape[::2]) if vnb.shape[1] == 0 else vnb.mean(dim=1)
        self.decay_rate = decay_rate
        self.evnt_record = [] if jump_type == "simulate" else evnt_record
        self.backtrace = []

        for net in [self.Fc, self.Fh, self.Gc, self.Gh, self.L, self.A]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.1)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, t, u):
        # print(t)
        c = u[:, :, :self.p]
        h = u[:, :, self.p:]

        h_ = self.A(torch.cat((h, torch.stack(tuple(self.aggregate_func(h[:, list(self.graph.neighbors(nid)), :])
                                                    for nid in self.graph.nodes()), dim=1)), dim=2))

        u_ = torch.cat((c, h_), dim=2)
        dc = -self.Fc(u_) * c + self.Gc(u_)
        dh = -self.Fh(u_) * h

        # ensure the gradient of c is orthogonal to the current c (trajectory on a sphere)
        # dc = dc - (torch.sum(dc * c, dim=2, keepdim=True) / torch.sum(c * c, dim=2, keepdim=True)) * c
        # add decay at constant rate
        dc = dc - self.decay_rate * c

        return torch.cat((dc, dh), dim=2)

    def simulate_jump(self, t0, t1, u0, u1):
        assert t0 < t1
        du = torch.zeros(u0.shape)
        sequence = []

        if self.jump_type == "simulate":
            lmbda_dt = (self.L(u0) + self.L(u1)) / 2 * (t1 - t0)
            rd = torch.rand(lmbda_dt.shape)
            dN = torch.zeros(lmbda_dt.shape)
            dN[rd < lmbda_dt ** 2 / 2] += 1
            dN[rd < lmbda_dt ** 2 / 2 + lmbda_dt * torch.exp(-lmbda_dt)] += 1

            du[:, :, self.p:] += self.Gh(u1) * dN
            for evnt in dN.nonzero():
                for _ in range(dN[tuple(evnt)].int()):
                    sequence.append((t1,) + tuple(evnt))
            self.evnt_record.extend(sequence)

        return du

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
        return t

    def read_jump(self, t1, u1):
        du = torch.zeros(u1.shape)

        if self.jump_type == "read":
            inf = sys.maxsize
            lid = bisect.bisect_left(self.evnt_record, (t1, -inf, -inf, -inf))
            rid = bisect.bisect_right(self.evnt_record, (t1, inf, inf, inf))

            dN = torch.zeros(u1.shape[:2] + (self.q,))
            for evnt in self.evnt_record[lid:rid]:
                t, sid, nid, eid = evnt
                dN[sid, nid, eid] += 1

            du[:, :, self.p:] += self.Gh(u1) * dN

        return du


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
                plt.plot(tsave.numpy(), dat, linewidth=0.7)

            # plot the state function (backward trace)
            if (tsave_ is not None) and (trace_ is not None):
                for dat in list(trace_[:, sid, nid, :].detach().numpy().T):
                    plt.plot(tsave_.numpy(), dat, linewidth=0.3, linestyle="dashed", color="black")

            # plot the intensity function
            plt.plot(tsave.numpy(), lmbda[:, sid, nid, :].detach().numpy(), linewidth=2.0)
            if (grid is not None) and (lmbda_real is not None):
                plt.plot(grid.numpy(), lmbda_real[sid], linewidth=1.0)

            tsne_current = [record for record in tsne if (record[1] == sid and record[2] == nid)]
            evnt_time = np.array([tsave[record[0]] for record in tsne_current])
            evnt_type = np.array([record[3] for record in tsne_current])

            plt.scatter(evnt_time, np.ones(len(evnt_time)) * 7.0, 2.0, c=evnt_type)
            plt.savefig(args.dataset + '/{:03d}_{:03d}_{:04d}'.format(batch_id[sid], nid, itr) + appendix, dpi=250)
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


def forward_pass(func, u0, tspan, dt, batch):
    # merge the sequences to create a sequence
    batch_record = sorted([(record[0],) + (sid,) + record[1:]
                           for sid in range(len(batch)) for record in batch[sid]])

    # set up grid
    tsave, gtid, evnt_record, tsne = create_tsave(tspan[0], tspan[1], dt, batch_record, args.evnt_align)
    func.evnt_record = evnt_record

    # forward pass
    trace = odeint(func, u0.repeat(len(batch), 1, 1), tsave, method='jump_adams', rtol=1.0e-6, atol=1.0e-8)
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

    p, q, dt, tspan = 5, 1, 0.05, (0.0, 100.0)
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
    func = ODEFunc(p, q, jump_type=args.jump_type, graph=G)
    if args.restart:
        checkpoint = torch.load(args.dataset + "/" + args.paramr)
        func.load_state_dict(checkpoint['func_state_dict'])
        u0p = checkpoint['u0p']
        u0q = checkpoint['u0q']
        it0 = checkpoint['it0']
    else:
        u0p = torch.randn(G.number_of_nodes(), p, requires_grad=True)
        u0q = torch.zeros(G.number_of_nodes(), q)
        it0 = 0

    optimizer = optim.Adam([{'params': func.parameters()},
                            {'params': u0p, 'lr': 1e-2},
                            {'params': u0q}
                            ], lr=1e-3, weight_decay=1e-4)

    loss_meter = RunningAverageMeter()

    # if read from history, then fit to maximize likelihood
    it = it0
    if func.jump_type == "read":
        while it < args.niters:
            # clear out gradients for variables
            optimizer.zero_grad()

            # sample a mini-batch, create a grid based on that
            batch_id = np.random.choice(len(TSTR), args.batch_size, replace=False)
            batch = [TSTR[seqid] for seqid in batch_id]

            # forward pass
            tsave, trace, lmbda, gtid, tsne, loss = forward_pass(func, torch.cat((u0p, u0q), dim=1), tspan, dt, batch)
            loss_meter.update(loss.item() / len(batch))
            print("iter: {}, running ave loss: {:.4f}".format(it, loss_meter.avg))

            # backward prop
            func.backtrace.clear()
            loss.backward()

            # step
            optimizer.step()

            it = it+1

            # validate and visualize
            if it % args.nsave == 0:
                # use the full validation set for forward pass
                tsave, trace, lmbda, gtid, tsne, loss = forward_pass(func, torch.cat((u0p, u0q), dim=1), tspan, dt, TSVA)
                print("iter: {}, validation loss: {:.4f}".format(it, loss.item()/len(TSVA)))

                # backward prop
                func.backtrace.clear()
                loss.backward()

                # visualize
                tsave_ = torch.tensor([record[0] for record in reversed(func.backtrace)])
                trace_ = torch.stack(tuple(record[1] for record in reversed(func.backtrace)))
                visualize(tsave, trace, lmbda, tsave_, trace_, tsave[gtid], lmbda_va_real, tsne, range(len(TSVA)), it)

                # save
                torch.save({'func_state_dict': func.state_dict(), 'u0p': u0p, 'u0q': u0q, 'it0': it},
                           args.dataset + "/" + args.paramw)

    # simulate for validation set
    func.jump_type = "simulate"
    tsave, trace, lmbda, gitd, tsne, loss = forward_pass(func, torch.cat((u0p, u0q), dim=1), tspan, dt, [[]]*len(TSVA))
    visualize(tsave, trace, lmbda, None, None, None, None, tsne, range(len(TSVA)), it, "simulate")

    # computing testing error
    # func.jump_type = "read"
    # tsave, trace, lmbda, gtid, tsne, loss = forward_pass(func, torch.cat((u0p, u0q), dim=1), tspan, dt, TSTE)
    # print("iter: {}, testing loss: {:.4f}".format(it, loss.item()/len(TSTE)))
