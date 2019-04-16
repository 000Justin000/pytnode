import sys
import subprocess
import signal
import argparse
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torchdiffeq import odeint_adjoint as odeint
from utils import RunningAverageMeter, ODEJumpFunc, create_outpath


signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

parser = argparse.ArgumentParser('point_processes')
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--jump_type', type=str, default='none')
parser.add_argument('--paramr', type=str, default='params.pth')
parser.add_argument('--paramw', type=str, default='params.pth')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--nsave', type=int, default=10)
parser.add_argument('--dataset', type=str, default='exponential_hawkes')
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
                plt.plot(tsave.numpy(), dat, linewidth=0.3)

            # plot the state function (backward trace)
            if (tsave_ is not None) and (trace_ is not None):
                for dat in list(trace_[:, sid, nid, :].detach().numpy().T):
                    plt.plot(tsave_.numpy(), dat, linewidth=0.2, linestyle="dotted", color="black")

            # plot the intensity function
            if (grid is not None) and (lmbda_real is not None):
                plt.plot(grid.numpy(), lmbda_real[sid], linewidth=1.0, color="gray")
            plt.plot(tsave.numpy(), lmbda[:, sid, nid, :].detach().numpy(), linewidth=0.7, color="red")

            tsne_current = [record for record in tsne if (record[1] == sid and record[2] == nid)]
            evnt_time = np.array([tsave[record[0]] for record in tsne_current])
            evnt_type = np.array([record[3] for record in tsne_current])

            plt.scatter(evnt_time, np.ones(len(evnt_time)) * 7.0, 2.0, c=evnt_type)
            plt.savefig(outpath + '/{:03d}_{:03d}_{:04d}'.format(batch_id[sid], nid, itr) + appendix, dpi=250)
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
    # t(ime)s(equence)n(ode)e(vent)
    batch_record = sorted([(record[0],) + (sid,) + record[1:]
                           for sid in range(len(batch)) for record in batch[sid]])

    # set up grid
    tsave, gtid, evnt_record, tsne = create_tsave(tspan[0], tspan[1], dt, batch_record, args.evnt_align)
    func.set_evnts(evnts=evnt_record)

    # forward pass
    trace = odeint(func, z0.repeat(len(batch), 1, 1), tsave, method='jump_adams', rtol=1.0e-5, atol=1.0e-7)
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
    # write all parameters to output file
    print(args, flush=True)

    # fix seeding for randomness
    if args.seed0:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    # create a graph
    G = nx.Graph()
    G.add_node(0)

    dim_c, dim_h, dim_N, dt, tspan = 3, 2, 1, 0.05, (0.0, 100.0)
    path = "./data/point_processes/"
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
    func = ODEJumpFunc(dim_c, dim_h, dim_N, dim_hidden=20, num_hidden=0, jump_type=args.jump_type, evnt_align=args.evnt_align, activation=nn.CELU(), graph=G)
    if args.restart:
        checkpoint = torch.load(outpath + "/" + args.paramr)
        func.load_state_dict(checkpoint['func_state_dict'])
        c0 = checkpoint['c0']
        h0 = checkpoint['h0']
        it0 = checkpoint['it0']
    else:
        c0 = torch.randn(G.number_of_nodes(), dim_c, requires_grad=True)
        h0 = torch.zeros(G.number_of_nodes(), dim_h)
        it0 = 0

    optimizer = optim.Adam([{'params': func.parameters()},
                            {'params': c0},
                            ], lr=1e-3, weight_decay=1e-5)

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
            tsave, trace, lmbda, gtid, tsne, loss = forward_pass(func, torch.cat((c0, h0), dim=1), tspan, dt, batch)
            loss_meter.update(loss.item() / len(batch))
            print("iter: {}, running ave loss: {:.4f}".format(it, loss_meter.avg), flush=True)

            # backward prop
            func.backtrace.clear()
            loss.backward()

            # step
            optimizer.step()

            it = it+1

            # validate and visualize
            if it % args.nsave == 0:
                # use the full validation set for forward pass
                tsave, trace, lmbda, gtid, tsne, loss = forward_pass(func, torch.cat((c0, h0), dim=1), tspan, dt, TSVA)
                print("iter: {}, validation loss: {:.4f}".format(it, loss.item()/len(TSVA)), flush=True)

                # backward prop
                func.backtrace.clear()
                loss.backward()

                # visualize
                tsave_ = torch.tensor([record[0] for record in reversed(func.backtrace)])
                trace_ = torch.stack(tuple(record[1] for record in reversed(func.backtrace)))
                visualize(tsave, trace, lmbda, tsave_, trace_, tsave[gtid], lmbda_va_real, tsne, range(len(TSVA)), it)

                # save
                torch.save({'func_state_dict': func.state_dict(), 'c0': c0, 'h0': h0, 'it0': it}, outpath + '/' + args.paramw)


    # simulate for validation set
    func.set_evnts(jump_type="simulate")
    tsave, trace, lmbda, gitd, tsne, loss = forward_pass(func, torch.cat((c0, h0), dim=1), tspan, dt, [[]]*len(TSVA))
    visualize(tsave, trace, lmbda, None, None, None, None, tsne, range(len(TSVA)), it, "simulate")

    # computing testing error
    func.set_evnts(jump_type="read")
    tsave, trace, lmbda, gtid, tsne, loss = forward_pass(func, torch.cat((c0, h0), dim=1), tspan, dt, TSTE)
    print("iter: {}, testing loss: {:.4f}".format(it, loss.item()/len(TSTE)), flush=True)
