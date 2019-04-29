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
parser.set_defaults(restart=False, evnt_align=False, seed0=False, debug=False)
parser.add_argument('--restart', dest='restart', action='store_true')
parser.add_argument('--evnt_align', dest='evnt_align', action='store_true')
parser.add_argument('--seed0', dest='seed0', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
args = parser.parse_args()

outpath = create_outpath('stack_overflow')
commit = subprocess.check_output("git log --pretty=format:\'%h\' -n 1", shell=True).decode()
if not args.debug:
    matplotlib.use('agg')
    sys.stdout = open(outpath + '/' + commit + '.log', 'w')
    sys.stderr = open(outpath + '/' + commit + '.err', 'w')


def read_stackoverflow(scale=1.0, h_dt=0.0, t_dt=0.0):
    time_seqs = []
    with open('./data/stack_overflow/time.txt') as ftime:
        seqs = ftime.readlines()
        for seq in seqs:
            time_seqs.append([float(t) for t in seq.split()])

    tmin = min([min(seq) for seq in time_seqs])
    tmax = max([max(seq) for seq in time_seqs])

    mark_seqs = []
    with open('./data/stack_overflow/event.txt') as fmark:
        seqs = fmark.readlines()
        for seq in seqs:
            mark_seqs.append([int(k) for k in seq.split()])

    m2mid = {m: mid for mid, m in enumerate(np.unique(sum(mark_seqs, [])))}

    evnt_seqs = [[((h_dt+time-tmin)*scale, m2mid[mark]) for time, mark in zip(time_seq, mark_seq)] for time_seq, mark_seq in zip(time_seqs, mark_seqs)]
    random.shuffle(evnt_seqs)

    return evnt_seqs, (0.0, ((tmax+t_dt)-(tmin-h_dt))*scale)


if __name__ == '__main__':
    # write all parameters to output file
    print(args, flush=True)

    # fix seeding for randomness
    if args.seed0:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    dim_c, dim_h, dim_N, dt = 10, 10, 22, 1.0/30.0
    TS, tspan = read_stackoverflow(1.0/30.0/24.0/3600.0, 1.0, 1.0)
    nseqs = len(TS)

    TSTR, TSVA, TSTE = TS[:int(nseqs*0.85)], TS[int(nseqs*0.85):int(nseqs*0.9)], TS[int(nseqs*0.9):]

    # initialize / load model
    func = ODEJumpFunc(dim_c, dim_h, dim_N, dim_N, dim_hidden=32, num_hidden=2, ortho=True, jump_type=args.jump_type, evnt_align=args.evnt_align, activation=nn.CELU())
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
            tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, batch, args.evnt_align)
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
                tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, TSVA, args.evnt_align)

                # backward prop
                func.backtrace.clear()
                loss.backward()
                print("iter: {}, validation loss: {:10.4f}, type error: {}".format(it, loss.item()/len(TSVA), mete), flush=True)

                # visualize
                tsave_ = torch.tensor([record[0] for record in reversed(func.backtrace)])
                trace_ = torch.stack(tuple(record[1] for record in reversed(func.backtrace)))
                visualize(outpath, tsave, trace, lmbda, tsave_, trace_, None, None, tsne, range(len(TSVA)), it)

                # save
                torch.save({'func_state_dict': func.state_dict(), 'c0': c0, 'h0': h0, 'it0': it, 'optimizer_state_dict': optimizer.state_dict()}, outpath + '/' + args.paramw)


    # computing testing error
    tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, TSTE, args.evnt_align)
    visualize(outpath, tsave, trace, lmbda, None, None, None, None, tsne, range(len(TSTE)), it, "testing")
    print("iter: {}, testing loss: {:10.4f}, type error: {}".format(it, loss.item()/len(TSTE), mete), flush=True)

    # simulate events
    func.jump_type="simulate"
    tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, [[]]*10, args.evnt_align)
    visualize(outpath, tsave, trace, lmbda, None, None, None, None, tsne, range(10), it, "simulate")
