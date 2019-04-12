import sys
import signal
import argparse
import random
import numpy as np
import bisect
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import gc

import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torchdiffeq import odeint
from utils import MLP, GCU, RNN, RunningAverageMeter

parser = argparse.ArgumentParser('coupled_osciallators')
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--paramr', type=str, default='params.pth')
parser.add_argument('--paramw', type=str, default='params.pth')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--nsave', type=int, default=10)
parser.add_argument('--num_validation', type=int, default=100)
parser.add_argument('--dataset', type=str, default='three_body')
parser.add_argument('--suffix', type=str, default='')
parser.set_defaults(restart=False, debug=False)
parser.add_argument('--restart', dest='restart', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
args = parser.parse_args()


class COFunc(nn.Module):

    def __init__(self, p, graph=None, m=1.0, k=1.0):
        super(COFunc, self).__init__()

        self.p = p
        self.setup_graph(graph, m, k)

    def setup_graph(self, graph=None, m=1.0, k=1.0):

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

        self.e2id = {tuple(sorted(e)): idx for idx, e in enumerate(self.graph.edges())}

    def forward(self, t, u):
        v = u[:, :, :self.p]
        r = u[:, :, self.p:]

        nbk = lambda nid: self.k[[self.e2id[tuple(sorted((nid, nbid)))] for nbid in self.graph.neighbors(nid)]]

        dv = torch.stack(tuple(((r[:, list(self.graph.neighbors(nid)), :] - r[:, [nid], :]) * nbk(nid).reshape(1, -1, 1)).sum(dim=1) / self.m[nid]
                               for nid in self.graph.nodes()), dim=1)
        dr = v

        # KE = (0.5 * self.m * (v**2).sum(dim=2)).sum(dim=1)
        # KV = (0.5 * self.k * torch.stack(tuple(((r[:, e[0], :] - r[:, e[1], :])**2).sum(dim=1) for e in self.graph.edges), dim=1)).sum(dim=1)
        # print('energy @ {0:6.2f} is: '.format(t), KE+KV)

        return torch.cat((dv, dr), dim=2)


class ODEFunc(nn.Module):

    def __init__(self, dim_z, dim_hidden=20, num_hidden=0, activation=nn.CELU(), graph=None, aggregation=None):
        super(ODEFunc, self).__init__()

        self.F = GCU(dim_z, 0, dim_hidden, num_hidden, activation, aggregation)
        self.setup_graph(graph)

    def setup_graph(self, graph=None):

        if graph:
            self.graph = graph
        else:
            self.graph = nx.Graph()
            self.graph.add_node(0)

    def forward(self, t, z):
        assert len(z.shape) == 3, 'z need to be 3 dimensional vector accessed by [seq_id, node_id, dim_id]'

        dz = torch.stack(tuple(self.F(z[:, nid, :], z[:, list(self.graph.neighbors(nid)), :]) for nid in self.graph.nodes()), dim=1)

        # orthogonalize dc w.r.t. to c
        # dz = dz - (dz*z).sum(dim=2, keepdim=True) / (z*z).sum(dim=2, keepdim=True) * z

        return dz


def cotrace(cofunc, num_seqs, tsave):

    # set initial states
    v0 = torch.randn(num_seqs, cofunc.graph.number_of_nodes(), cofunc.p)
    v0 = v0 - v0.mean(dim=1, keepdim=True)
    r0 = torch.randn(num_seqs, cofunc.graph.number_of_nodes(), cofunc.p)

    trajs = odeint(cofunc, torch.cat((v0, r0), dim=2), tsave, method='adams', rtol=1.0e-5, atol=1.0e-7)

    return trajs


def log_normal_pdf(x, mean, logvar):
    const = torch.log(torch.tensor(2.0 * np.pi))
    return -0.5 * (const + logvar + (x - mean) ** 2.0 / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.0
    lstd2 = lv2 / 2.0

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.0) / (2.0 * v2)) - 0.5
    return kl


def visualize(trace, it=0, num_seqs=sys.maxsize, appendix=""):
    for sid in range(min(num_seqs, trace.shape[1])):
        for tid in range(0, trace.shape[0], 5):
            fig = plt.figure(figsize=(6, 6), facecolor='white')
            axe = plt.gca()
            axe.set_title('Coupled Oscillators')
            axe.set_xlabel('x')
            axe.set_ylabel('y')
            axe.set_xlim(-10.0, 10.0)
            axe.set_ylim(-10.0, 10.0)

            plt.scatter(trace[tid, sid, :, 0].detach().numpy(), trace[tid, sid, :, 1].detach().numpy(), c=range(trace.shape[2]))

            plt.savefig(args.dataset + args.suffix + '/{:06d}_{:03d}_{:04d}'.format(it, sid, tid) + appendix, dpi=250)
            fig.clf()
            plt.close(fig)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

    # fix seeding for randomness
    if args.debug:
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

    # num_seqs : number of validation examples
    num_seqs, dim_p, dim_z, dim_hidden, dt, tspan = 500, 2, 5, 20, 0.05, (-10.0, 20.0)

    # set up the grid
    tsave = torch.arange(tspan[0], tspan[1], dt)
    nts = (tsave < 0).sum()

    # set up the coupled-oscillator function, update the graph
    G0 = nx.complete_graph(10)
    cofunc = COFunc(dim_p)
    cofunc.setup_graph(G0)

    # simulate the validation trace
    trajs_va = cotrace(cofunc, num_seqs, tsave)
    visualize(trajs_va[nts:, :, :, dim_p:dim_p*2], it=0, num_seqs=3)

    # define encoder and decoder networks
    func = ODEFunc(dim_z, dim_hidden=20, num_hidden=0, activation=nn.CELU())
    enc = RNN(dim_p, dim_z*2, dim_hidden, 0, nn.Tanh())
    dec = MLP(dim_z, dim_p, dim_hidden, 1, nn.CELU())

    # set up the optimizer
    optimizer = optim.Adam([{'params': func.parameters()},
                            {'params': enc.parameters()},
                            {'params': dec.parameters()},
                            ], lr=1e-4, weight_decay=1e-6)

    # initialize / load model
    if args.restart:
        checkpoint = torch.load(args.dataset + args.suffix + "/" + args.paramr)
        func.load_state_dict(checkpoint['func_state_dict'])
        enc.load_state_dict(checkpoint['enc_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        it0 = checkpoint['it0']
    else:
        it0 = 0

    def compute_loss(trajs, visualization=False, it=0, appendix=""):
        
        if (not visualization) and ((it != 0) or (appendix != "")):
            print("Warning: appendix is ignored when visualization is false")

        # compute the encoding using trajectory upto t=0.0
        out = enc(trajs[:nts, :, :, dim_p:dim_p*2])
        qz0_mean, qz0_logvar = out[-1, :, :, :dim_z], out[-1, :, :, dim_z:]
        epsilon = torch.randn(qz0_mean.shape)

        z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
        pred_z = odeint(func, z0, tsave[nts:])
        pred_x = dec(pred_z)

        if visualization:
            visualize(pred_x, it=it, num_seqs=3, appendix=appendix)

        # compute loss
        noise_std = torch.zeros(pred_x.shape) + 0.3
        logpx = log_normal_pdf(trajs[nts:, :, :, dim_p:dim_p*2], pred_x, noise_std).sum()
        pz0_mean, pz0_logvar = torch.zeros(z0.shape), torch.zeros(z0.shape)
        kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum()

        # update parameters
        loss = (-logpx + kl) / (trajs.shape[1] * trajs.shape[2])
        
        return loss

    loss_meter = RunningAverageMeter()

    it = it0
    while it < args.niters:
        # clear out gradients for variables
        optimizer.zero_grad()

        # first sample the number of particles, then sample the trace
        num_vertices = random.randint(1, 6)
        G = nx.complete_graph(num_vertices)
        cofunc.setup_graph(G)
        func.setup_graph(G)
        trajs_tr = cotrace(cofunc, args.batch_size, tsave)

        # compute the loss and go down the gradient
        loss = compute_loss(trajs_tr)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())
        print('Iter: {}, running avg elbo: {:.4f}'.format(it, -loss_meter.avg), flush=True)

        it += 1

        # validate and visualize
        if it % args.nsave == 0:

            func.setup_graph(G0)
            loss = compute_loss(trajs_va, visualization=True, it=it)

            print('Iter: {}, validation elbo: {:.4f}'.format(it, -loss.item()), flush=True)

            # save
            torch.save({'func_state_dict': func.state_dict(),
                        'enc_state_dict':  enc.state_dict(),
                        'dec_state_dict':  dec.state_dict(),
                        'optimizer_state_dict':  optimizer.state_dict(),
                        'it0': it}, args.dataset + args.suffix + '/' + args.paramw)
        gc.collect()

    # compute validation loss again in the end
    func.setup_graph(G0)
    loss = compute_loss(trajs_va, visualization=True, it=it)
    print('Iter: {}, validation elbo: {:.4f}'.format(it, -loss.item()), flush=True)
