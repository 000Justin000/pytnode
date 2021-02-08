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
from modules import RunningAverageMeter, ODEJumpFunc
from utils import forward_pass, visualize, create_outpath
from sklearn import mixture
from earthquake_utils import EarthquakeGenerator


signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

parser = argparse.ArgumentParser('tweet')
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

outpath = create_outpath("earthquake")
commit = subprocess.check_output("git log --pretty=format:\'%h\' -n 1", shell=True).decode()
if not args.debug:
    matplotlib.use('agg')
    sys.stdout = open(outpath + '/' + commit + '.log', 'w')
    sys.stderr = open(outpath + '/' + commit + '.err', 'w')

def visualize_(outpath, tsave, gtid, lmbda, gsmean, gsvar, events, itr):
    for i in range(len(gtid)-1):
        events_current = np.array([np.array(evnt[1]) for evnt in events
                                   if (tsave[gtid[i]] < evnt[0] < tsave[gtid[i+1]])])

        gaussian_weight = lmbda[gtid[i], 0, :].detach().numpy()
        gaussian_center = gsmean[gtid[i], 0, :, :].detach().numpy()
        gaussian_var = gsvar[gtid[i], 0, :, :].detach().numpy()

        x, y = np.meshgrid(np.linspace(-0.5, 0.5, 500), np.linspace(-0.5, 0.5, 500))
        density = np.zeros(x.shape)
        for gs_weight, gs_center, gs_var in zip(gaussian_weight, gaussian_center, gaussian_var):
            gs_pdf = np.exp(-0.5*((x-gs_center[0])**2.0/gs_var[0] +
                                  (y-gs_center[1])**2.0/gs_var[1])) / ((2*np.pi) * np.sqrt(gs_var[0] * gs_var[1]))
            density += gs_weight * gs_pdf

        fig = plt.figure(figsize=(6, 6), facecolor='white')
        axe = plt.gca()
        axe.set_title('Earthquakes')
        axe.set_xlabel('longitude')
        axe.set_ylabel('latitude')
        axe.set_xlim(-0.5, 0.5)
        axe.set_ylim(-0.5, 0.5)

        cs = plt.contour(x, y, density, levels=[2**j for j in range(-10, 10)])
        plt.clabel(cs, inline=1, fontsize=10)

        if len(events_current) != 0:
            plt.scatter(events_current[:, 0], events_current[:, 1], 3.0, c="red")
        plt.scatter(gaussian_center[:, 0], gaussian_center[:, 1], 3.0, c="orange")

        plt.savefig(outpath + '/{:04d}_{:04d}.png'.format(itr, i), dpi=250)
        fig.clf()
        plt.close(fig)

def estimate_density(events, tspan):
    if args.seed0:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    X = np.array([np.array(evnt[1]) for evnt in events])
    clf = mixture.GaussianMixture(n_components=5, covariance_type="diag")
    clf.fit(X)

    gaussian_weight = clf.weights_ * (len(events) / tspan[1])
    gaussian_center = clf.means_
    gaussian_var = clf.covariances_

    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 500), np.linspace(-0.5, 0.5, 500))
    density = np.zeros(x.shape)
    for gs_weight, gs_center, gs_var in zip(gaussian_weight, gaussian_center, gaussian_var):
        gs_pdf = np.exp(-0.5*((x-gs_center[0])**2.0/gs_var[0] +
                              (y-gs_center[1])**2.0/gs_var[1])) / ((2*np.pi) * np.sqrt(gs_var[0] * gs_var[1]))
        density += gs_weight * gs_pdf

    fig = plt.figure(figsize=(6, 6), facecolor='white')
    axe = plt.gca()
    axe.set_title('Earthquakes')
    axe.set_xlabel('longitude')
    axe.set_ylabel('latitude')
    axe.set_xlim(-0.5, 0.5)
    axe.set_ylim(-0.5, 0.5)

    cs = plt.contour(x, y, density, levels=[2**j for j in range(-10, 10)])
    plt.clabel(cs, inline=1, fontsize=10)

    if len(X) != 0:
        plt.scatter(X[:, 0], X[:, 1], 3.0, c="red")
    plt.scatter(gaussian_center[:, 0], gaussian_center[:, 1], 3.0, c="orange")

    plt.savefig(outpath + '/baseline.svg', dpi=250)
    fig.clf()
    plt.close(fig)

    return (gaussian_weight, gaussian_center, gaussian_var)


if __name__ == '__main__':
    # write all parameters to output file
    print(args, flush=True)

    # fix seeding for randomness
    if args.seed0:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    dim_c, dim_h, dim_N, dim_E, dt = 10, 10, 5, 2, 1.0/52.0
    tspan_tr = (0.0, 35.12)
    tspan_va = (0.0, 49.41)
    eg = EarthquakeGenerator()

    # initialize / load model
    func = ODEJumpFunc(dim_c, dim_h, dim_N, dim_E, dim_hidden=32, num_hidden=1, jump_type=args.jump_type, evnt_align=args.evnt_align, activation=nn.Tanh(), ortho=True, evnt_embedding="continuous")
    c0 = torch.randn(dim_c)
    h0 = torch.zeros(dim_h)
    it0 = 0

#   optimizer = optim.LBFGS(func.parameters(), max_iter=20)
    optimizer = optim.AdamW(func.parameters(), lr=1e-3, weight_decay=2.5e-4)

    if args.restart:
        checkpoint = torch.load(args.paramr)
        func.load_state_dict(checkpoint['func_state_dict'])
        c0 = checkpoint['c0']
        h0 = checkpoint['h0']
        it0 = checkpoint['it0']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # if read from history, then fit to maximize likelihood
    it = it0 - (1 if args.restart else 0)
    if func.jump_type == "read":
        while it < args.niters:
            def closure():
                # clear out gradients for variables
                optimizer.zero_grad()

                # random initial
                c0 = torch.randn(dim_c)

                # generate event sequences
                TSTR = eg.event_seqs(num_seqs=args.batch_size, radius=0.2, scale=1.0/52.0/7.0/24.0/3600.0, time_cutoff=35.12)

                # forward pass
                tsave, trace, lmbda, gtid, tsne, loss, mete, gsmean, gsvar = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan_tr, dt, TSTR, args.evnt_align)

                # backward prop
                func.backtrace.clear()
                loss.backward()
                print("iter: {}, current loss: {:10.4f}".format(it, loss.item()), flush=True)

                return loss

            # step
            optimizer.step(closure)

            it = it+1

            if it % args.nsave == 0:
                # validation
                optimizer.zero_grad()

                # generate event sequence
                TSVA = eg.event_seqs(centers=[np.radians([37.229564, -120.047533])], num_seqs=1, radius=0.2, scale=1.0/52.0/7.0/24.0/3600.0, rand_rot=False)

                tsave, trace, lmbda, gtid, tsne, loss, mete, gsmean, gsvar = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan_va, dt, TSVA, args.evnt_align)

                func.backtrace.clear()
                loss.backward()
                print("iter: {}, validation loss: {:10.4f}, type error: {}".format(it, loss.item(), mete), flush=True)

                # visualize
                tsave_ = torch.tensor([record[0] for record in reversed(func.backtrace)])
                trace_ = torch.stack(tuple(record[1] for record in reversed(func.backtrace)))
                visualize_(outpath, tsave, gtid[0::52], lmbda, gsmean, gsvar, TSVA[0], it)
                visualize(outpath, tsave, trace, lmbda, tsave_, trace_, None, None, None, range(len(TSVA)), it)

                # save
                torch.save({'func_state_dict': func.state_dict(), 'c0': c0, 'h0': h0, 'it0': it, 'optimizer_state_dict': optimizer.state_dict()}, outpath + '/' + '{:04d}'.format(it) + args.paramw)
