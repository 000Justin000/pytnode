import os
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
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--restart', type=bool, default=False)
parser.add_argument('--paramr', type=str, default='r.pth')
parser.add_argument('--paramw', type=str, default='w.pth')
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
        self.Z = nn.Sequential(nn.Linear(p+q, p),   nn.Tanh())
        self.L = nn.Sequential(nn.Linear(p+q, p+q), nn.Sigmoid(), nn.Linear(p+q, q), nn.Softplus())
        self.TS = TS

        for net in [self.F, self.G, self.Z, self.L]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.1)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, t, u):
        du = -self.F(u) * u + self.G(u) * torch.cat((self.Z(u), torch.tensor([TS.intensity(t)])), 0)
        return du - torch.dot(du, u) / torch.dot(u, u) * u


def visualize(tsave, trace, lmbda, envt, itr):
    fig = plt.figure(figsize=(6, 6), facecolor='white')
    axe = plt.gca()
    axe.set_title('Point Process Modeling')
    axe.set_xlabel('time')
    axe.set_ylabel('intensity')
    axe.set_ylim(-3.0, 3.0)
    for dat in list(trace.detach().numpy().T):
        plt.plot(tsave.numpy(), dat, linewidth=0.5)
    plt.plot(tsave.numpy(), lmbda.detach().numpy(), linewidth=2.0)
    plt.scatter(envt.numpy(), np.ones(len(envt))*2.0)
    fig.savefig('figs/{:03d}'.format(itr), dpi=150)
    # fig.show()


if __name__ == '__main__':
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
    else:
        u0p = torch.randn(p, requires_grad=True)
        u0q = torch.zeros(q)

    grid = torch.arange(tspan[0], tspan[1], dt)
    envt = torch.tensor([ts[idx][i] for ts in TS.tss for idx in ts for i in range(len(ts[idx])) if tspan[0] < ts[idx][i] < tspan[1]])
    tsave, pos = torch.sort(torch.cat((grid, envt)))
    _, od = torch.sort(pos)
    gridid = od[:len(grid)]
    envtid = od[len(grid):]

    optimizer = optim.Adam(itertools.chain(func.parameters(), [u0p, u0q]), lr=1e-3, weight_decay=1e-4)

    try:
        for i in range(100):
            optimizer.zero_grad()
            trace = odeint(func, torch.cat((u0p, u0q)), tsave, method='adams')
            lmbda = func.L(trace)
            loss = -((torch.log(lmbda[envtid, 0])).sum() - (lmbda[gridid, 0] * dt).sum())
            loss.backward()
            optimizer.step()
            visualize(tsave, trace, lmbda, envt, i)
            print(loss)
    except KeyboardInterrupt:
        torch.save({'func_state_dict': func.state_dict(), 'u0p': u0p, 'u0q': u0q}, args.paramw)
