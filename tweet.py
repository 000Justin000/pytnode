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
from modules import RunningAverageMeter, ODEJumpFunc
from utils import forward_pass, visualize, create_outpath


signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

parser = argparse.ArgumentParser('point_processes')
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--jump_type', type=str, default='none')
parser.add_argument('--paramr', type=str, default='params.pth')
parser.add_argument('--paramw', type=str, default='params.pth')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--nsave', type=int, default=10)
parser.add_argument('--dataset', type=str, default='politics')
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


def read_twitter(scale=1.0):
    if args.dataset == 'politics':
        dat = np.loadtxt('./data/tweet/Sentiment_PoliticsTwitter.txt')
    elif args.dataset == 'movie':
        dat = np.loadtxt('./data/tweet/Sentiment_MovieTwitter.txt')
    elif args.dataset == 'fight':
        dat = np.loadtxt('./data/tweet/Sentiment_FightTwitter.txt')
    elif args.dataset == 'bollywood':
        dat = np.loadtxt('./data/tweet/Sentiment_BollywoodTwitter.txt')

    od = np.argsort(np.array([tuple(el[:2]) for el in dat], dtype=[('x', 'i'), ('y', 'f')]))
    dat = dat[od, :]

    uid = np.unique(dat[:, 0])

    time = dat[:, 1] * scale
    event = dat[:, 2]

    tmin = time.min()
    tmax = time.max()

    edges = np.searchsorted(dat[:, 0], np.append(0, uid)+0.5)[1:-1]
    tseqs = np.split(time, edges)
    kseqs = np.split(event, edges)

    evnt_seqs = [[(t-tmin, [k]) for t, k in zip(tseq, kseq)] for tseq, kseq in zip(tseqs, kseqs)]
    random.shuffle(evnt_seqs)

    return evnt_seqs, (0.0, tmax-tmin)


def running_ave(TSTR, TSTE, type_forecast):
    stmt = []
    for ts in TSTR:
        stmt.extend([evnt[1][0] for evnt in ts])
    stmt0 = sum(stmt)/len(stmt)

    et_error = []
    for ts in TSTE:
        time = [-np.inf] + [evnt[0] for evnt in ts]
        stmt = [stmt0] + [evnt[1][0] for evnt in ts]
        runave = np.cumsum(stmt) / np.arange(1, len(stmt)+1)
        for evnt in ts:
            type_preds = np.zeros(len(type_forecast))
            for tid, t in enumerate(type_forecast):
                loc = np.searchsorted(time, evnt[0]-t)-1
                assert loc >= 0
                type_preds[tid] = runave[loc]
            et_error.append((type_preds - evnt[1][0])**2.0)
    print(sum(et_error)/len(et_error))


if __name__ == '__main__':
    # write all parameters to output file
    print(args, flush=True)

    # fix seeding for randomness
    if args.seed0:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    dim_c, dim_h, dim_N, dim_E, dt = 5, 5, 1, 1, 1.0/24.0
    TS, tspan = read_twitter(1.0/24.0/3600.0)
    nseqs = len(TS)

    TSTR, TSVA, TSTE = TS[:int(nseqs*0.8)], TS[int(nseqs*0.8):int(nseqs*0.9)], TS[int(nseqs*0.9):]

    running_ave(TSTR, TSTE, [1.0/24.0 * i for i in range(0, 11, 2)])

    # initialize / load model
    func = ODEJumpFunc(dim_c, dim_h, dim_N, dim_E, dim_hidden=20, num_hidden=2, jump_type=args.jump_type, evnt_align=args.evnt_align, activation=nn.Tanh(), ortho=True, evnt_embedding="continuous")
    c0 = torch.randn(dim_c, requires_grad=True)
    h0 = torch.zeros(dim_h)
    it0 = 0
    optimizer = optim.Adam([{'params': func.parameters()},
                            {'params': c0, 'lr': 1.0e-2},
                            ], lr=1e-3, weight_decay=1e-5)

    if args.restart:
        checkpoint = torch.load(args.paramr)
        func.load_state_dict(checkpoint['func_state_dict'])
        c0 = checkpoint['c0']
        h0 = checkpoint['h0']
        it0 = checkpoint['it0']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
            tsave, trace, lmbda, gtid, tsne, loss, mete, gsmean = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, batch, args.evnt_align, [1.0/24.0 * i for i in range(0, 11, 2)])
            loss_meter.update(loss.item() / len(batch))

            # backward prop
            func.backtrace.clear()
            loss.backward()
            print("iter: {}, current loss: {:10.4f}, running ave loss: {:10.4f}, type error: {}".format(it, loss.item()/len(batch), loss_meter.avg, mete), flush=True)

            # step
            optimizer.step()

            it = it+1

            # validate and visualize
            if it % args.nsave == 0:
                # use the full validation set for forward pass
                tsave, trace, lmbda, gtid, tsne, loss, mete, gsmean = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, TSVA, args.evnt_align, [1.0/24.0 * i for i in range(0, 11, 2)])

                # backward prop
                func.backtrace.clear()
                loss.backward()
                print("iter: {}, validation loss: {:10.4f}, type error: {}".format(it, loss.item()/len(TSVA), mete), flush=True)

                # visualize
                tsave_ = torch.tensor([record[0] for record in reversed(func.backtrace)])
                trace_ = torch.stack(tuple(record[1] for record in reversed(func.backtrace)))
                visualize(outpath, tsave, trace, lmbda, tsave_, trace_, tsave, [gsmean[:, i, :, 0].detach().numpy() * 10.0 for i in range(len(TSVA))], tsne, range(len(TSVA)), it)

                # save
                torch.save({'func_state_dict': func.state_dict(), 'c0': c0, 'h0': h0, 'it0': it, 'optimizer_state_dict': optimizer.state_dict()}, outpath + '/' + args.paramw)


    # computing testing error
    tsave, trace, lmbda, gtid, tsne, loss, mete, gsmean = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, TSTE, args.evnt_align, [1.0/24.0 * i for i in range(0, 11, 2)])
    visualize(outpath, tsave, trace, lmbda, None, None, tsave, [gsmean[:, i, :, 0].detach().numpy() * 10.0 for i in range(len(TSTE))], tsne, range(len(TSTE)), it, "testing")
    print("iter: {}, testing loss: {:10.4f}, type error: {}".format(it, loss.item()/len(TSTE), mete), flush=True)

    # simulate events
    func.jump_type="simulate"
    tsave, trace, lmbda, gtid, tsne, loss, mete, gsmean = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, [[]]*10, args.evnt_align, [1.0/24.0 * i for i in range(0, 11, 2)])
    visualize(outpath, tsave, trace, lmbda, None, None, tsave, [gsmean[:, i, :, 0].detach().numpy() * 10.0 for i in range(10)], tsne, range(10), it, "simulate")
