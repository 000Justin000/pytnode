import os
import argparse
import time
import numpy as np
import functools
import tick.dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint

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
        print(t)
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
        self.F = nn.Sequential(nn.Linear(p+q, p+q), nn.Sigmoid())
        self.G = nn.Sequential(nn.Linear(p+q, p+q), nn.Sigmoid())
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


if __name__ == '__main__':
    p, q, dt, sigma, tspan = 5, 1, 0.05, 0.10, (0.0, 100.0)
    dat = tick.dataset.fetch_hawkes_bund_data()
    TS = TimeSeries(sum(dat, [])[0:q])

    func = ODEFunc(p, q, TS)
    u0 = torch.cat((torch.randn(p), torch.zeros(q)))
    grid = torch.arange(tspan[0], tspan[1], dt)
    tsave, pos = torch.sort(torch.cat((grid, torch.tensor([ts[idx][i] for ts in TS.tss for idx in ts for i in range(len(ts[idx])) if tspan[0] < ts[idx][i] < tspan[1]]))))
    _, od = torch.sort(pos)
    gridid = od[:len(grid)]
    envtid = od[len(grid):]

    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    optimizer.zero_grad()
    trace = odeint(func, u0, tsave)
    lmbda = func.L(trace)
    loss = -((torch.log(lmbda[envtid, 0])).sum() - (lmbda[gridid, 0] * dt).sum())
    print("backward")
    loss.backward()
    optimizer.step()