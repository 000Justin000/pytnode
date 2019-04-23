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
import networkx as nx
from utils import RunningAverageMeter, ODEJumpFunc, create_outpath
from utils import forward_pass, visualize


signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

parser = argparse.ArgumentParser('point_processes')
parser.add_argument('--niters', type=int, default=100)
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

outpath = create_outpath('book_order')
commit = subprocess.check_output("git log --pretty=format:\'%h\' -n 1", shell=True).decode()
if not args.debug:
    matplotlib.use('agg')
    sys.stdout = open(outpath + '/' + commit + '.log', 'w')
    sys.stderr = open(outpath + '/' + commit + '.err', 'w')


def read_orderbook():
    from scipy.interpolate import interp1d

    msg_full = np.loadtxt('./data/order_book/AMZN_2012-06-21_34200000_57600000_message_1.csv', delimiter=',')
    odb_full = np.loadtxt('./data/order_book/AMZN_2012-06-21_34200000_57600000_orderbook_1.csv', delimiter=',')

    def price_interp(msg, odb):
        x = msg[:, 0] - 34200.0
        y = (odb[:, 0]+odb[:, 2])/2.0
        _, uid = np.unique(x, return_index=True)
        return interp1d(x[uid], y[uid], bounds_error=False, fill_value=(y[0], y[-1]))  # fix starting here

    cs_full = price_interp(msg_full, odb_full)

    edges = np.searchsorted(msg_full[:, 0], np.linspace(34200, 57600, num=391, endpoint=True))[1:-1]

    msgs = np.split(msg_full, edges)

    xx = np.linspace(0.0, 60.0, num=1201, endpoint=True)

    def evnt_at(i_msg):
        i, msg = i_msg
        evnt_seq = [(record[0] - (34200.0 + i*60.0), 0, round(float(record[5]+1.0)/2.0))
                    for record in msg if record[1] == 1.0]
        return evnt_seq

    evnt_seqs = list(map(evnt_at, enumerate(msgs)))

    def interp_at(i):
        y = cs_full(i*60.0+xx)
        y = (y/y[0] - 1.0)*100.0
        return interp1d(xx, y)

    pctc_seqs = list(map(interp_at, range(390)))

    return evnt_seqs, pctc_seqs


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

    dim_c, dim_h, dim_N, dt, tspan = 5, 5, 2, 0.05, (0.0, 60.0)
    TS, PC = read_orderbook()

    TSTR, TSVA, TSTE = TS[:300], TS[300:345], TS[345:]
    PCTR, PCVA, PCTE = PC[:300], PC[300:345], PC[345:]

    # initialize / load model
    func = ODEJumpFunc(dim_c, dim_h, dim_N, dim_hidden=20, num_hidden=0, jump_type=args.jump_type, evnt_align=args.evnt_align, activation=nn.CELU(), graph=G)
    if args.restart:
        checkpoint = torch.load(args.paramr)
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
            tsave, trace, lmbda, gtid, tsne, loss = forward_pass(func, torch.cat((c0, h0), dim=1), tspan, dt, batch, args.evnt_align)
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
                tsave, trace, lmbda, gtid, tsne, loss = forward_pass(func, torch.cat((c0, h0), dim=1), tspan, dt, TSVA, args.evnt_align)
                print("iter: {}, validation loss: {:.4f}".format(it, loss.item()/len(TSVA)), flush=True)

                # backward prop
                func.backtrace.clear()
                loss.backward()

                # visualize
                tsave_ = torch.tensor([record[0] for record in reversed(func.backtrace)])
                trace_ = torch.stack(tuple(record[1] for record in reversed(func.backtrace)))
                visualize(outpath, tsave, trace, lmbda, tsave_, trace_, None, None, tsne, range(len(TSVA)), it)

                # save
                torch.save({'func_state_dict': func.state_dict(), 'c0': c0, 'h0': h0, 'it0': it}, outpath + '/' + args.paramw)


    # computing testing error
    tsave, trace, lmbda, gtid, tsne, loss = forward_pass(func, torch.cat((c0, h0), dim=1), tspan, dt, TSTE, args.evnt_align)
    visualize(outpath, tsave, trace, lmbda, None, None, None, None, tsne, range(len(TSTE)), it, "testing")
    print("iter: {}, testing loss: {:.4f}".format(it, loss.item()/len(TSTE)), flush=True)

    # simulate events
    func.set_evnts(jump_type="simulate")
    tsave, trace, lmbda, gtid, tsne, loss = forward_pass(func, torch.cat((c0, h0), dim=1), tspan, dt, [[]]*10, args.evnt_align)
    visualize(outpath, tsave, trace, lmbda, None, None, None, None, tsne, range(10), it, "simulate")
