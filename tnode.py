import os
import sys
import signal
import argparse
import time
import random
import numpy as np
import functools
import itertools
import bisect
import matplotlib.pyplot as plt
import tick.dataset

import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torchdiffeq import odeint_adjoint as odeint

parser = argparse.ArgumentParser('tnode')
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--jump_type', type=str, default='none')
parser.add_argument('--paramr', type=str, default='param.pth')
parser.add_argument('--paramw', type=str, default='param.pth')
parser.add_argument('--path', type=str, default='figs/')
parser.add_argument('--batch_size', type=int, default=1)
parser.set_defaults(restart=False, evnt_approx=False)
parser.add_argument('--restart', dest='restart', action='store_true')
parser.add_argument('--evnt_approx', dest='evnt_approx', action='store_true')
args = parser.parse_args()


class ODEFunc(nn.Module):

    def __init__(self, p, q, jump_type="simulate", evnt_record=None, graph=None, aggregate_func=None):
        super(ODEFunc, self).__init__()

        assert jump_type in ["simulate", "read", "none"], "invalide jump_type, must be one of [simulate, read, none]"
        self.p = p
        self.q = q
        self.jump_type = jump_type
        self.Fc = nn.Sequential(nn.Linear(p + q, p), nn.Softplus())
        self.Fh = nn.Sequential(nn.Linear(p + q, q), nn.Softplus())
        self.Gc = nn.Sequential(nn.Linear(p + q, p), nn.Softplus())
        self.Gh = nn.Sequential(nn.Linear(p + q, q), nn.Softplus())
        self.Z = nn.Sequential(nn.Linear(p + q, p), nn.Tanh())
        self.L = nn.Sequential(nn.Linear(p + q, q), nn.Softplus())
        self.A = nn.Sequential(nn.Linear(2 * q, q), nn.Softplus())
        self.evnt_record = [] if jump_type == "simulate" else evnt_record
        self.backtrace = []
        if graph:
            self.graph = graph
        else:
            self.graph = nx.Graph()
            self.graph.add_node(0)
        if aggregate_func:
            self.aggregate_func = aggregate_func
        else:
            self.aggregate_func = lambda vnb: torch.zeros(vnb.shape[::2]) if vnb.shape[1] == 0 else vnb.mean(dim=1)

        for net in [self.Fc, self.Fh, self.Gc, self.Gh, self.Z, self.L, self.A]:
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
        dc = -self.Fc(u_) * c + self.Gc(u_) * self.Z(u_)
        dh = -self.Fh(u_) * h

        # ensure the gradient of c is orthogonal to the current c (trajectory on a sphere)
        dc = dc - (torch.sum(dc * c, dim=2, keepdim=True) / torch.sum(c * c, dim=2, keepdim=True)) * c

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


def read_timeseries(dat, num_vertices=1):
    timeseries = []
    for u in range(num_vertices):
        for t in dat[0][u]:
            timeseries.append((t, u, 0))

    return [sorted(timeseries), sorted(timeseries), sorted(timeseries)]


# this function takes in a time series and create a grid for modeling it
# it takes an array of sequences of three tuples, and extend it to four tuple
def create_tsave(TS, dt, tmin, tmax, evnt_approx=False):
    """
    :param TS: an array of sequences of three tuples
    :param dt: step size
    :param tmin: min time of sequence
    :param tmax: max time of the sequence
    :param evnt_approx: whether to round the event time up to the next grid point
    :return tsave: the time to save state in ODE simulation
    :return TS_record: an array of four tuples, the 1st element is the sequence ID, 2nd element is the time stamp ID
    """

    if evnt_approx:
        tc = lambda t: np.ceil(t / dt) * dt
    else:
        tc = lambda t: t

    grid = np.arange(tmin, tmax+dt, dt)
    evnt = np.array([tc(record[0]) for TS_ in TS for record in TS_ if tmin < tc(record[0]) < tmax])
    tsave = np.sort(np.unique(np.concatenate((grid, evnt))))
    t2tid = {t: tid for tid, t in enumerate(tsave)}

    # g(rid)tid
    # t(ime)s(equence)n(ode)e(vent)
    gtid = [t2tid[t] for t in grid]
    tsne = [(t2tid[tc(record[0])],) + (sid,) + record[1:]
            for sid in range(len(TS)) for record in TS[sid]
            if tmin < tc(record[0]) < tmax]

    return torch.tensor(tsave), gtid, tsne


def visualize(tsave, trace, tsave_, trace_, lmbda, tsne, batch_id, itr):
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

            evnt_time, evnt_type = np.array(tuple(zip(*[(tsave[record[0]], record[3]) for record in tsne
                                                        if (record[1] == sid and record[2] == nid)])))
            plt.scatter(evnt_time, np.ones(len(evnt_time)) * 7.0, 2.0, c=evnt_type)
            plt.savefig(args.path + '{:03d}_{:03d}_{:03d}.pdf'.format(batch_id[sid], nid, itr), dpi=150)
            fig.clf()
            plt.close(fig)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

    # create a graph
    G = nx.Graph()
    # G.add_node(0)
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    p, q, dt, sigma, tspan = 5, 1, 0.05, 0.10, (60.0, 70.0)
    dat = tick.dataset.fetch_hawkes_bund_data()
    TS = read_timeseries(dat, G.number_of_nodes())

    # initialize / load model
    torch.manual_seed(0)
    func = ODEFunc(p, q, jump_type=args.jump_type, graph=G)
    if args.restart:
        checkpoint = torch.load(args.paramr)
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

    # if read from history, then fit to maximize likelihood
    it = it0
    if func.jump_type == "read":
        while it < args.niters:
            optimizer.zero_grad()

            # sample a mini-batch, create a grid based on that
            batch_id = np.random.choice(len(TS), args.batch_size, replace=False)
            batch = [TS[sid] for sid in batch_id]
            tsave, gtid, tsne = create_tsave(batch, dt, tspan[0], tspan[1], args.evnt_approx)

            # merge the sequences to create a sequence
            evnt_record = sorted([(record[0],) + (sid,) + record[1:]
                                  for sid in range(len(batch)) for record in batch[sid]])
            func.evnt_record = evnt_record

            # forward pass
            trace = odeint(func, torch.cat((u0p, u0q), dim=1).repeat(args.batch_size, 1, 1), tsave, method='jump_adams')
            lmbda = func.L(trace)
            loss = -(sum([torch.log(lmbda[record]) for record in tsne]) - (lmbda[gtid, :, :, :] * dt).sum())

            # backward prop
            func.backtrace.clear()
            loss.backward()
            optimizer.step()
            tsave_ = torch.tensor([record[0] for record in reversed(func.backtrace)])
            trace_ = torch.stack(tuple(record[1] for record in reversed(func.backtrace)))
            visualize(tsave, trace, tsave_, trace_, lmbda, tsne, batch_id, it)
            print("iter: ", it, "    loss: ", loss)

            it = it + 1
            torch.save({'func_state_dict': func.state_dict(), 'u0p': u0p, 'u0q': u0q, 'it0': it}, args.paramw)

    # simulate trace
    tsave, _, _ = create_tsave([], dt, tspan[0], tspan[1])
    trace = odeint(func, torch.cat((u0p, u0q), dim=1).repeat(args.batch_size, 1, 1), tsave, method='jump_adams')
    lmbda = func.L(trace)
    visualize(tsave, trace, None, None, lmbda, [], it - 1)
