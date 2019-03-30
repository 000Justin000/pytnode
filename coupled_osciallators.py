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
from torchdiffeq import odeint


class COFunc(nn.Module):

    def __init__(self, p, graph=None, m=1.0, k=1.0):
        super(COFunc, self).__init__()

        self.p = p
        if graph:
            self.graph = graph

            if isinstance(m, list):
                assert len(m) == self.graph.number_of_nodes()
                self.m = m
            else:
                self.m = torch.ones(self.graph.number_of_nodes()) * m

            if isinstance(k, list):
                assert len(k) == self.graph.number_of_edges()
                self.k = k
            else:
                self.k = torch.ones(self.graph.number_of_edges()) * m
        else:
            assert not isinstance(m, list), 'must provide graph when providing node weights'
            assert not isinstance(k, list), 'must provide graph when providing edge weights'
            self.graph = nx.Graph()
            self.graph.add_node(0)
            self.m = torch.ones(1) * m
            self.k = torch.ones(0) * k

        self.e2id = {tuple(sorted(e)): idx for idx, e in enumerate(graph.edges())}

    def forward(self, t, u):
        v = u[:, :, :self.p]
        r = u[:, :, self.p:]

        nbk = lambda nid: self.k[[self.e2id[tuple(sorted((nid, nbid)))] for nbid in self.graph.neighbors(nid)]]

        dv = torch.stack(tuple(((r[:, list(self.graph.neighbors(nid)), :] - r[:, [nid], :]) * nbk(nid).reshape(1, -1, 1)).sum(dim=1) / self.m[nid]
                               for nid in self.graph.nodes()), dim=1)
        dr = v

        KE = (0.5 * self.m * (v**2).sum(dim=2)).sum(dim=1)
        KV = (0.5 * self.k * torch.stack(tuple(((r[:, e[0], :] - r[:, e[1], :])**2).sum(dim=1) for e in self.graph.edges), dim=1)).sum(dim=1)

        print('energy @ {0:6.2f} is: '.format(t), KE+KV)

        return torch.cat((dv, dr), dim=2)


def visualize(trace, appendix=""):
    for sid in range(trace.shape[1]):
        for tid in range(trace.shape[0]):
            fig = plt.figure(figsize=(6, 6), facecolor='white')
            axe = plt.gca()
            axe.set_title('Coupled Oscillators')
            axe.set_xlabel('x')
            axe.set_ylabel('y')
            axe.set_xlim(-6.0, 6.0)
            axe.set_ylim(-6.0, 6.0)

            plt.scatter(trace[tid, sid, :, 2].numpy(), trace[tid, sid, :, 3].numpy(), c=range(trace.shape[2]))

            plt.savefig('tmp/{:03d}_{:04d}'.format(sid, tid) + appendix, dpi=250)
            fig.clf()
            plt.close(fig)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

    # create a graph
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    nseq, p, dt, tspan = 5, 2, 0.05, (0.0, 100.0)

    # initialize / load model
    torch.manual_seed(1)
    func = COFunc(p, graph=G)

    v0 = torch.zeros(nseq, G.number_of_nodes(), p)
    r0 = torch.randn(nseq, G.number_of_nodes(), p)

    tsave = torch.arange(tspan[0], tspan[1], dt)

    trace = odeint(func, torch.cat((v0, r0), dim=2), tsave, method='adams', rtol=1.0e-7, atol=1.0e-9)

    visualize(trace)

