import os
import sys
import signal
import argparse
import time
import random
import numpy as np
import functools
import itertools
import matplotlib.pyplot as plt
import tick.dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint

parser = argparse.ArgumentParser('tnode')
parser.set_defaults(restart=False)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--restart', dest='restart', action='store_true')
parser.add_argument('--paramr', type=str, default='param.pth')
parser.add_argument('--paramw', type=str, default='param.pth')
parser.add_argument('--path', type=str, default='figs/')
args = parser.parse_args()

class TimeSeries:

    def __init__(self, series, sigma=0.1):
        self.tmin = min(functools.reduce(lambda x, y: np.concatenate((x, y)), series))
        self.twd = sigma * 10
        self.sigma = sigma
        self.tss = []
        for se in series:
            ts = {}
            for t in se:
                idx = int(np.floor((t - self.tmin) / self.twd))
                if idx not in ts:
                    ts[idx] = []
                ts[idx].append(t)
            self.tss.append(ts)

    def intensity(self, t):
        assert not torch.isnan(t)
        if torch.isinf(t): return torch.zeros(len(self.tss))

        vv = torch.zeros(len(self.tss))
        idx = int(np.floor(t.detach().numpy() - self.tmin) / self.twd)
        for (i, ts) in enumerate(self.tss):
            tt = torch.tensor(ts.get(idx-1, []) + ts.get(idx, []) + ts.get(idx+1, []))
            vv[i] = torch.exp(-0.5 * ((tt - t) / self.sigma)**2 / (self.sigma * np.sqrt(2*np.pi))).sum()
        return vv


class ODEFunc(nn.Module):

    def __init__(self, p, q, TS):
        super(ODEFunc, self).__init__()

        self.p = p
        self.q = q
        self.F = nn.Sequential(nn.Linear(p+q, p+q), nn.Softplus())
        self.G = nn.Sequential(nn.Linear(p+q, p+q), nn.Softplus())
        self.Z = nn.Sequential(nn.Linear(p+q, p), nn.Tanh())
        self.L = nn.Sequential(nn.Linear(p+q, q), nn.Softplus())
        self.TS = TS

        for net in [self.F, self.G, self.Z, self.L]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.1)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, t, u):
        du = -self.F(u) * u + self.G(u) * torch.cat((self.Z(u), torch.tensor([TS.intensity(t)])), 0)
        dup = du[:self.p] - torch.dot(du[:self.p], u[:self.p]) / torch.dot(u[:self.p], u[:self.p]) * u[:self.p]
        duq = du[self.p:]
        return torch.cat((dup, duq))


def visualize(tsave, trace, lmbda, envt, itr):
    plt.figure(figsize=(6, 6), facecolor='white')
    axe = plt.gca()
    axe.set_title('Point Process Modeling')
    axe.set_xlabel('time')
    axe.set_ylabel('intensity')
    axe.set_ylim(-5.0, 5.0)
    for dat in list(trace.detach().numpy().T):
        plt.plot(tsave.numpy(), dat, linewidth=0.5)
    plt.plot(tsave.numpy(), lmbda.detach().numpy(), linewidth=1.0)
    plt.scatter(envt.numpy(), np.ones(len(envt))*2.0)
    plt.savefig(args.path + '{:03d}'.format(itr), dpi=150)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

    p, q, dt, sigma, tspan = 5, 1, 0.05, 0.10, (0.0, 300.0)
    dat = tick.dataset.fetch_hawkes_bund_data()
    TS = TimeSeries(sum(dat, [])[0:q])

    # initialize / load model
    func = ODEFunc(p, q, TS)
    if args.restart:
        checkpoint = torch.load(args.paramr)
        func.load_state_dict(checkpoint['func_state_dict'])
        u0p = checkpoint['u0p']
        u0q = checkpoint['u0q']
        it0 = checkpoint['it0']
    else:
        u0p = torch.randn(p, requires_grad=True)
        u0q = torch.zeros(q)
        it0 = 0

    grid = torch.arange(tspan[0], tspan[1], dt)
    envt = torch.tensor([ts[idx][i] for ts in TS.tss for idx in ts for i in range(len(ts[idx]))
                         if tspan[0] < ts[idx][i] < tspan[1]])
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
        trace = odeint(func, torch.cat((u0p, u0q)), tsave, method='adams')
        lmbda = func.L(trace)
        loss = -((torch.log(lmbda[envtid, 0])).sum() - (lmbda[gridid, 0] * dt).sum())
        loss.backward()
        optimizer.step()
        visualize(tsave, trace, lmbda, envt, i)
        torch.save({'func_state_dict': func.state_dict(), 'u0p': u0p, 'u0q': u0q, 'it0': i+1}, args.paramw)
        print("iter: ", i, "    loss: ", loss)
