import sys
import subprocess
import signal
import argparse
import random
import numpy as np
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from utils import RunningAverageMeter, RNN, SoftPlus, create_outpath
from utils import poisson_lmbda, exponential_hawkes_lmbda, powerlaw_hawkes_lmbda, self_inhibiting_lmbda, visualize


signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

parser = argparse.ArgumentParser('point_processes')
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--paramr', type=str, default='params.pth')
parser.add_argument('--paramw', type=str, default='params.pth')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--nsave', type=int, default=10)
parser.add_argument('--dataset', type=str, default='poisson')
parser.add_argument('--suffix', type=str, default='')

parser.set_defaults(restart=False, evnt_align=False, seed0=False, debug=False)
parser.add_argument('--restart', dest='restart', action='store_true')
parser.add_argument('--evnt_align', dest='evnt_align', action='store_true')
parser.add_argument('--seed0', dest='seed0', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
args = parser.parse_args()

outpath = create_outpath(args.dataset)
commit = subprocess.check_output("git log --pretty=format:\'%h\' -n 1", shell=True).decode()
if not args.debug:
    matplotlib.use('agg')
    sys.stdout = open(outpath + '/' + commit + '.log', 'w')
    sys.stderr = open(outpath + '/' + commit + '.err', 'w')


def read_timeseries(filename, num_seqs=sys.maxsize):
    with open(filename) as f:
        seqs = f.readlines()[:num_seqs]
    return [[(float(t), vid, 0) for vid, vts in enumerate(seq.split(";")) for t in vts.split()] for seq in seqs]


def forward_pass(func, h0, tspan, dt, batch):
    evnts = sorted([(record[0],) + (sid,) + record[1:] for sid in range(len(batch)) for record in batch[sid]])

    tc = lambda t: np.round(np.ceil((t-tspan[0]) / dt) * dt + tspan[0], decimals=8)
    grid = np.round(np.arange(tspan[0], tspan[1]+dt, dt), decimals=8)
    t2tid = {t: tid for tid, t in enumerate(grid)}
    evnts_vec = torch.zeros(len(grid), len(batch), 1, func.dim_in)
    for sid, seq in enumerate(batch):
        for evnt in seq:
            evnts_vec[t2tid[tc(evnt[0])], sid, evnt[1], evnt[2]] += 1.0

    L = SoftPlus()
    lmbda = L(func(evnts_vec, h0=h0.repeat(len(batch), 1, 1))[:-1])

    loss = -(sum([torch.log(lmbda[(t2tid[tc(evnt[0])],) + evnt[1:]]) for evnt in evnts]) - (lmbda*dt).sum())

    return torch.tensor(grid), lmbda, loss


if __name__ == '__main__':
    # write all parameters to output file
    print(args, flush=True)

    # fix seeding for randomness
    if args.seed0:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    dim_h, dim_N, dt, tspan = 20, 1, 0.05, (0.0, 100.0)
    path = "./data/point_processes/"
    TSTR = read_timeseries(path + args.dataset + "_training.csv")
    TSVA = read_timeseries(path + args.dataset + "_validation.csv")
    TSTE = read_timeseries(path + args.dataset + "_testing.csv")

    if args.dataset == "poisson":
        lmbda_va_real = poisson_lmbda(tspan[0], tspan[1], dt, 0.2, TSVA)
        lmbda_te_real = poisson_lmbda(tspan[0], tspan[1], dt, 0.2, TSTE)
    elif args.dataset == "exponential_hawkes":
        lmbda_va_real = exponential_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 1.0, TSVA, args.evnt_align)
        lmbda_te_real = exponential_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 1.0, TSTE, args.evnt_align)
    elif args.dataset == "powerlaw_hawkes":
        lmbda_va_real = powerlaw_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 2.0, 1.0, TSVA, args.evnt_align)
        lmbda_te_real = powerlaw_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 2.0, 1.0, TSTE, args.evnt_align)
    elif args.dataset == "self_inhibiting":
        lmbda_va_real = self_inhibiting_lmbda(tspan[0], tspan[1], dt, 0.5, 0.2, TSVA, args.evnt_align)
        lmbda_te_real = self_inhibiting_lmbda(tspan[0], tspan[1], dt, 0.5, 0.2, TSTE, args.evnt_align)

    # initialize / load model
    func = RNN(dim_N, dim_N, dim_hidden=dim_h, num_hidden=1, activation=nn.Tanh())
    if args.restart:
        checkpoint = torch.load(args.paramr)
        func.load_state_dict(checkpoint['func_state_dict'])
        h0 = checkpoint['h0']
        it0 = checkpoint['it0']
    else:
        h0 = torch.randn(1, dim_h, requires_grad=True)
        it0 = 0

    optimizer = optim.Adam([{'params': func.parameters()},
                            {'params': h0, 'lr': 1.0e-2},
                            ], lr=1e-3, weight_decay=1e-5)

    loss_meter = RunningAverageMeter()

    # if read from history, then fit to maximize likelihood
    it = it0
    while it < args.niters:
        # clear out gradients for variables
        optimizer.zero_grad()

        # sample a mini-batch, create a grid based on that
        batch_id = np.random.choice(len(TSTR), args.batch_size, replace=False)
        batch = [TSTR[seqid] for seqid in batch_id]

        # forward pass
        tsave, lmbda, loss = forward_pass(func, h0, tspan, dt, batch)
        loss_meter.update(loss.item() / len(batch))
        print("iter: {}, running ave loss: {:.4f}".format(it, loss_meter.avg), flush=True)

        # backward prop
        loss.backward()

        # step
        optimizer.step()

        it = it+1

        # validate and visualize
        if it % args.nsave == 0:
            # use the full validation set for forward pass
            tsave, lmbda, loss = forward_pass(func, h0, tspan, dt, TSVA)
            print("iter: {}, validation loss: {:.4f}".format(it, loss.item()/len(TSVA)), flush=True)
            visualize(outpath, tsave, None, lmbda, None, None, tsave, lmbda_va_real, None, range(len(TSVA)), it)

            # save
            torch.save({'func_state_dict': func.state_dict(), 'h0': h0, 'it0': it}, outpath + '/' + args.paramw)


    # computing testing error
    tsave, lmbda, loss = forward_pass(func, h0, tspan, dt, TSTE)
    print("iter: {}, testing loss: {:.4f}".format(it, loss.item()/len(TSTE)), flush=True)
    visualize(outpath, tsave, None, lmbda, None, None, tsave, lmbda_te_real, None, range(len(TSTE)), it, "testing")
