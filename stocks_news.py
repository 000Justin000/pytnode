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
from torchdiffeq import odeint_adjoint as odeint
from utils import visualize, create_outpath, create_tsave, logsumexp


signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

parser = argparse.ArgumentParser('stock_news')
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

outpath = create_outpath("stocks_news")
commit = subprocess.check_output("git log --pretty=format:\'%h\' -n 1", shell=True).decode()
if not args.debug:
    matplotlib.use('agg')
    sys.stdout = open(outpath + '/' + commit + '.log', 'w')
    sys.stderr = open(outpath + '/' + commit + '.err', 'w')


def read_stock_news(scale=1.0):
    stocks = np.loadtxt('./data/stocks_news/stocks/stocks')
    stocks = stocks[np.logical_and(1136073600 < stocks[:, 0], stocks[:, 1] < 1546300800), :]

    time_intervals = stocks[:, [0, 1]]
    adjusted_price = (stocks[:, 3] / stocks[:, 2]) / (stocks[:, -1] / stocks[:, -2])

    events = np.loadtxt('./data/stocks_news/news/events_manual')

    edges = np.searchsorted(events[:, 0], [-np.inf] + list(time_intervals.reshape(-1)) + [np.inf])[1:-1]
    eseqs = np.split(events, edges)[1::2]

    evnt_seqs = [[((evnt[0]-time_interval[0])*scale, [evnt[1]]) for evnt in eseq] for time_interval, eseq in zip(time_intervals, eseqs)]

    random.shuffle(evnt_seqs)

    return evnt_seqs, adjusted_price, (0.0, 102.5*3600 * scale)


def forward_pass(func, z0, tspan, dt, batch, evnt_align, outcomes, rtol=1.0e-7, atol=1.0e-9):
    # merge the sequences to create a sequence
    evnts_raw = sorted([(evnt[0],) + (sid,) + evnt[1:] for sid in range(len(batch)) for evnt in batch[sid]])

    # set up grid
    tsave, gtid, evnts, tse = create_tsave(tspan[0], tspan[1], dt, evnts_raw, evnt_align)
    func.evnts = evnts

    # forward pass
    trace = odeint(func, z0.repeat(len(batch), 1), tsave, method='jump_adams', rtol=rtol, atol=atol)
    params = func.L(trace)
    lmbda = params[..., :func.dim_N]

    log_likelihood = 0
    loss0 = 0

    if func.evnt_embedding == "discrete":
        et_error = []
        for i in range(len(batch)):
            log_likelihood += torch.log(lmbda[-1, i, outcomes[i]] / sum(lmbda[-1, i, :]))
            type_pred = lmbda[-1, i].argmax().item()
            et_error.append((type_pred != outcomes[i]).float())

        METE = sum(et_error)/len(et_error)

    elif func.evnt_embedding == "continuous":
        gsmean = params[..., func.dim_N*(1+func.dim_E*0):func.dim_N*(1+func.dim_E*1)].view(params.shape[:-1]+(func.dim_N, func.dim_E))
        logvar = params[..., func.dim_N*(1+func.dim_E*1):func.dim_N*(1+func.dim_E*2)].view(params.shape[:-1]+(func.dim_N, func.dim_E))
        var = torch.exp(logvar)

        def log_normal_pdf(loc, k):
            const = torch.log(torch.tensor(2.0*np.pi))
            return -0.5*(const + logvar[loc] + (gsmean[loc] - func.evnt_embed(k))**2.0 / var[loc])

        et_error = []
        for i in range(len(batch)):
            log_gs = log_normal_pdf((-1, i), outcomes[i]).sum(dim=-1)  # debug
            log_likelihood += logsumexp(log_gs, dim=-1)
            mean_pred = ((lmbda[-1, i].view(func.dim_N, 1) * gsmean[-1, i]).sum(dim=0) / lmbda[-1, i].sum()).item()
            et_error.append((mean_pred - func.evnt_embed(outcomes[i])).norm(dim=-1)**2.0)
            loss0 += ((lmbda[0, i].view(func.dim_N, 1) * gsmean[0, i]).sum(dim=0) / lmbda[0, i].sum() - 1.0)**2.0

        METE = np.sqrt(sum(et_error)/len(et_error))

    if func.evnt_embedding == "discrete":
        return tsave, trace, lmbda, gtid, tse, -log_likelihood, METE
    elif func.evnt_embedding == "continuous":
        return tsave, trace, lmbda, gtid, tse, -log_likelihood + loss0, METE, gsmean


if __name__ == '__main__':
    # write all parameters to output file
    print(args, flush=True)

    # fix seeding for randomness
    if args.seed0:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    dim_c, dim_h, dim_N, dim_E, dt = 8, 16, 1, 1, 1.0/24.0
    TS, OC, tspan = read_stock_news(1.0/24.0/3600.0)
    nseqs = len(TS)

    TSTR, TSVA, TSTE = TS[:int(nseqs*0.8)], TS[int(nseqs*0.8):int(nseqs*0.9)], TS[int(nseqs*0.9):]
    OCTR, OCVA, OCTE = OC[:int(nseqs*0.8)], OC[int(nseqs*0.8):int(nseqs*0.9)], OC[int(nseqs*0.9):]

    # initialize / load model
    func = ODEJumpFunc(dim_c, dim_h, dim_N, dim_E, dim_hidden=32, num_hidden=1, jump_type=args.jump_type, evnt_align=args.evnt_align, activation=nn.CELU(), ortho=True, evnt_embedding="continuous")
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
            batch_outcomes = [OCTR[seqid] for seqid in batch_id]

            # forward pass
            tsave, trace, lmbda, gtid, tsne, loss, mete, gsmean = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, batch, args.evnt_align, batch_outcomes)
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
                tsave, trace, lmbda, gtid, tsne, loss, mete, gsmean = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, TSVA, args.evnt_align, OCVA)

                # backward prop
                func.backtrace.clear()
                loss.backward()
                print("iter: {}, validation loss: {:10.4f}, type error: {}".format(it, loss.item()/len(TSVA), mete), flush=True)

                # visualize
                tsave_ = torch.tensor([record[0] for record in reversed(func.backtrace)])
                trace_ = torch.stack(tuple(record[1] for record in reversed(func.backtrace)))
                visualize(outpath, tsave, trace, lmbda, tsave_, trace_, tsave, [gsmean[:, i, :, 0].detach().numpy() * 5.0 for i in range(len(TSVA))], tsne, range(len(TSVA)), it)

                # save
                torch.save({'func_state_dict': func.state_dict(), 'c0': c0, 'h0': h0, 'it0': it, 'optimizer_state_dict': optimizer.state_dict()}, outpath + '/' + args.paramw)


    # computing testing error
    tsave, trace, lmbda, gtid, tsne, loss, mete, gsmean = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, TSTE, args.evnt_align, OCTE)
    visualize(outpath, tsave, trace, lmbda, None, None, tsave, [gsmean[:, i, :, 0].detach().numpy() * 5.0 for i in range(len(TSTE))], tsne, range(len(TSTE)), it, "testing")
    print("iter: {}, testing loss: {:10.4f}, type error: {}".format(it, loss.item()/len(TSTE), mete), flush=True)

    # simulate events
    func.jump_type="simulate"
    tsave, trace, lmbda, gtid, tsne, loss, mete, gsmean = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, [[]]*10, args.evnt_align, np.ones(10))
    visualize(outpath, tsave, trace, lmbda, None, None, tsave, [gsmean[:, i, :, 0].detach().numpy() * 5.0 for i in range(10)], tsne, range(10), it, "simulate")
