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
parser.set_defaults(restart=False)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--restart', dest='restart', action='store_true')
parser.add_argument('--paramr', type=str, default='param.pth')
parser.add_argument('--paramw', type=str, default='param.pth')
parser.add_argument('--path', type=str, default='figs/')
args = parser.parse_args()


class ODEFunc(nn.Module):

    def __init__(self, p, q, jump_type="simulate", time_series=None, graph=None, aggregate_func=None):
        super(ODEFunc, self).__init__()

        assert jump_type in ["simulate", "read", "none"], "invalide jump_type, must be one of [simulate, read, none]"
        self.p = p
        self.q = q
        self.jump_type = jump_type
        self.Fc = nn.Sequential(nn.Linear(p+q, p), nn.Softplus())
        self.Fh = nn.Sequential(nn.Linear(p+q, q), nn.Softplus())
        self.Gc = nn.Sequential(nn.Linear(p+q, p), nn.Softplus())
        self.Gh = nn.Sequential(nn.Linear(p+q, q), nn.Softplus())
        self.Z = nn.Sequential(nn.Linear(p+q, p), nn.Tanh())
        self.L = nn.Sequential(nn.Linear(p+q, q), nn.Softplus())
        self.A = nn.Sequential(nn.Linear(2*q, q), nn.ReLU())
        self.timeseries = [] if jump_type == "simulate" else time_series
        self.backtrace = []
        if graph:
            self.graph = graph
        else:
            self.graph = nx.Graph()
            self.graph.add_node(0)
        if aggregate_func:
            self.aggregate_func = aggregate_func
        else:
            self.aggregate_func = lambda vnbrs: torch.zeros(self.q) if vnbrs.shape[0] == 0 else vnbrs.mean(dim=0)

        for net in [self.Fc, self.Fh, self.Gc, self.Gh, self.Z, self.L, self.A]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.1)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, t, u):
        print(t)
        assert u.shape[0] == self.graph.number_of_nodes()
        c = u[:, :self.p]
        h = u[:, self.p:]
        h_ = torch.stack(tuple(self.A(torch.cat((h[i, :], self.aggregate_func(h[list(self.graph.neighbors(i)), :]))))
                                 for i in self.graph.nodes))

        u_ = torch.cat((c, h_), dim=1)
        dc = -self.Fc(u_) * c + self.Gc(u_) * self.Z(u_)
        dh = -self.Fh(u_) * h

        # ensure the gradient of c is orthogonal to the current c (trajectory on a sphere)
        dc = dc - (torch.sum(dc*c, dim=1, keepdim=True) / torch.sum(c*c, dim=1, keepdim=True)) * c

        return torch.cat((dc,  dh), dim=1)

    def simulate_jump(self, t0, t1, u0, u1):
        assert t0 < t1
        du = torch.zeros(u0.shape)
        sequence = []

        if self.jump_type == "simulate":
            lmbda_dt = (self.L(u0)+self.L(u1))/2 * (t1-t0)
            rd = torch.rand(lmbda_dt.shape)
            dN = torch.zeros(lmbda_dt.shape)
            dN[rd < lmbda_dt**2/2] += 1
            dN[rd < lmbda_dt**2/2 + lmbda_dt*torch.exp(-lmbda_dt)] += 1

            du[:, self.p:] += self.Gh(u1) * dN
            for nid, evntid in dN.nonzero():
                for _ in range(dN[nid, evntid].int()):
                    sequence.append((t1, nid, evntid))
            self.timeseries.extend(sequence)

        return du

    def next_jump(self, t0, t1):
        assert t0 != t1, "t0 can not equal t1"

        t = t1

        if self.jump_type == "read":
            if t0 < t1:  # forward
                idx = bisect.bisect_right(self.timeseries, (t0, sys.maxsize, sys.maxsize))
                if idx != len(self.timeseries):
                    t = min(t1, self.timeseries[idx][0])
            else:  # backward
                idx = bisect.bisect_left(self.timeseries, (t0, -sys.maxsize, -sys.maxsize))
                if idx > 0:
                    t = max(t1, self.timeseries[idx-1][0])

        return torch.tensor(t, dtype=torch.float64)

    def read_jump(self, t1, u1):
        du = torch.zeros((self.graph.number_of_nodes(), p+q))

        if self.jump_type == "read":
            lid = bisect.bisect_left(self.timeseries, (t1, -sys.maxsize, -sys.maxsize))
            rid = bisect.bisect_right(self.timeseries, (t1, sys.maxsize, sys.maxsize))

            dN = torch.zeros((self.graph.number_of_nodes(), q))
            for evnt in self.timeseries[lid:rid]:
                t, nid, envtid = evnt
                dN[nid, envtid] += 1

            du[:, self.p:] += self.Gh(u1) * dN

        return du


def read_timeseries(num_vertices=1):
    dat = tick.dataset.fetch_hawkes_bund_data()
    timeseries = []
    for u in range(num_vertices):
        for t in dat[0][u]:
            timeseries.append((t, u, 0))

    return sorted(timeseries)


def visualize(tsave, trace, tsave_, trace_, lmbda, envt, itr):
    for i in range(trace.shape[1]):
        plt.figure(figsize=(6, 6), facecolor='white')
        axe = plt.gca()
        axe.set_title('Point Process Modeling')
        axe.set_xlabel('time')
        axe.set_ylabel('intensity')
        axe.set_ylim(-5.0, 5.0)
        for dat in list(trace[:, i, :].detach().numpy().T):
            plt.plot(tsave.numpy(), dat, linewidth=0.7)
        for dat in list(trace_[:, i, :].detach().numpy().T):
            plt.plot(tsave_.numpy(), dat, linewidth=0.3, linestyle="dashed", color="black")
        plt.plot(tsave.numpy(), lmbda[:, i, :].detach().numpy(), linewidth=2.0)
        plt.scatter(envt.numpy(), np.ones(len(envt))*2.0, 3.0)
        plt.savefig(args.path + '{:03d}_{:03d}'.format(i, itr), dpi=150)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

    # create a graph
    G = nx.Graph()
    # G.add_node(0)
    G.add_edge(0, 1)
    # G.add_edge(1, 2)

    p, q, dt, sigma, tspan = 5, 1, 0.05, 0.10, (50.0, 70.0)
    TS = read_timeseries(G.number_of_nodes())

    # initialize / load model
    torch.manual_seed(0)
    func = ODEFunc(p, q, jump_type="read", time_series=TS, graph=G)
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

    grid = torch.arange(tspan[0], tspan[1], dt)
    # envt = torch.tensor([ts[0] for ts in TS if tspan[1] < ts[0] < tspan[1]])
    envt = torch.tensor([])
    tsave, pos = torch.sort(torch.cat((grid, envt)))
    _, od = torch.sort(pos)
    gridid = od[:len(grid)]
    envtid = od[len(grid):]

    optimizer = optim.Adam([{'params': func.parameters()},
                            {'params': u0p, 'lr': 1e-2},
                            {'params': u0q}
                            ], lr=1e-3, weight_decay=1e-4)

    for i in range(it0, args.niters):
        optimizer.zero_grad()
        trace = odeint(func, torch.cat((u0p, u0q), dim=1), tsave, method='jump_adams')
        lmbda = func.L(trace)
        loss = -((torch.log(lmbda[envtid, :, 0])).sum() - (lmbda[gridid, :, 0] * dt).sum())
        func.backtrace = []  # debug
        loss.backward()
        optimizer.step()
        tsave_ = torch.cat(tuple(record[0].reshape((1)) for record in reversed(func.backtrace)))
        trace_ = torch.stack(tuple(record[1] for record in reversed(func.backtrace)))
        visualize(tsave, trace, tsave_, trace_, lmbda, envt, i)
        torch.save({'func_state_dict': func.state_dict(), 'u0p': u0p, 'u0q': u0q, 'it0': i+1}, args.paramw)
        print("iter: ", i, "    loss: ", loss)
