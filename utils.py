import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint_adjoint as odeint

# create the outdir
def create_outpath(dataset):
    path = os.getcwd()
    pid = os.getpid()

    wsppath = os.path.join(path, 'workspace')
    if not os.path.isdir(wsppath):
        os.mkdir(wsppath)

    outpath = os.path.join(wsppath, 'dataset:'+dataset + '-' + 'pid:'+str(pid))
    assert not os.path.isdir(outpath), 'output directory already exist (process id coincidentally the same), please retry'
    os.mkdir(outpath)

    return outpath


def visualize(outpath, tsave, trace, lmbda, tsave_, trace_, grid, lmbda_real, tsne, batch_id, itr, appendix=""):
    for sid in range(lmbda.shape[1]):
        fig = plt.figure(figsize=(6, 6), facecolor='white')
        axe = plt.gca()
        axe.set_title('Point Process Modeling')
        axe.set_xlabel('time')
        axe.set_ylabel('intensity')
        axe.set_ylim(-10.0, 10.0)

        # plot the state function
        if (tsave is not None) and (trace is not None):
            for dat in list(trace[:, sid, :].detach().numpy().T):
                plt.plot(tsave.numpy(), dat, linewidth=0.3)

        # plot the state function (backward trace)
        if (tsave_ is not None) and (trace_ is not None):
            for dat in list(trace_[:, sid, :].detach().numpy().T):
                plt.plot(tsave_.numpy(), dat, linewidth=0.2, linestyle="dotted", color="black")

        # plot the intensity function
        if (grid is not None) and (lmbda_real is not None):
            plt.plot(grid.numpy(), lmbda_real[sid], linewidth=1.0, color="gray")
        plt.plot(tsave.numpy(), lmbda[:, sid, :].detach().numpy(), linewidth=0.7)

        if tsne is not None:
            tsne_current = [evnt for evnt in tsne if evnt[1] == sid]
            # continue...
            tevnt = np.array([tsave[evnt[0]] for evnt in tsne_current])
            kevnt = np.array([evnt[2] if not (type(evnt[2]) == list) else evnt[2][0] for evnt in tsne_current])
            plt.scatter(tevnt, kevnt * 7.0, 0.5)

        plt.savefig(outpath + '/{:03d}_{:04d}'.format(batch_id[sid], itr) + appendix, dpi=250)
        fig.clf()
        plt.close(fig)


# this function takes in a time series and create a grid for modeling it
# it takes an array of sequences of three tuples, and extend it to four tuple
def create_tsave(tmin, tmax, dt, evnts_raw, evnt_align=False):
    """
    :param tmin: min time of sequence
    :param tmax: max time of the sequence
    :param dt: step size
    :param evnts_raw: tuple (raw_time, ...)
    :param evnt_align: whether to round the event time up to the next grid point
    :return tsave: the time to save state in ODE simulation
    :return gtid: grid time id
    :return evnts: tuple (rounded_time, ...)
    :return tsne: tuple (event_time_id, ...)
    """

    if evnt_align:
        tc = lambda t: np.round(np.ceil((t-tmin) / dt) * dt + tmin, decimals=8)
    else:
        tc = lambda t: t

    evnts = [(tc(evnt[0]),) + evnt[1:] for evnt in evnts_raw if tmin < tc(evnt[0]) < tmax]

    tgrid = np.round(np.arange(tmin, tmax+dt, dt), decimals=8)
    tevnt = np.array([evnt[0] for evnt in evnts])
    tsave = np.sort(np.unique(np.concatenate((tgrid, tevnt))))
    t2tid = {t: tid for tid, t in enumerate(tsave)}

    # g(rid)tid
    # t(ime)s(equence)n(ode)e(vent)
    gtid = [t2tid[t] for t in tgrid]
    tsne = [(t2tid[evnt[0]],) + evnt[1:] for evnt in evnts]

    return torch.tensor(tsave), gtid, evnts, tsne


def forward_pass(func, z0, tspan, dt, batch, evnt_align, type_forecast=[0.0]):
    # merge the sequences to create a sequence
    evnts_raw = sorted([(evnt[0],) + (sid,) + evnt[1:] for sid in range(len(batch)) for evnt in batch[sid]])

    # set up grid
    tsave, gtid, evnts, tsne = create_tsave(tspan[0], tspan[1], dt, evnts_raw, evnt_align)
    func.evnts = evnts

    # convert to numpy array
    tsavenp = tsave.numpy()

    # forward pass
    trace = odeint(func, z0.repeat(len(batch), 1), tsave, method='jump_adams', rtol=1.0e-5, atol=1.0e-7)
    params = func.L(trace)
    lmbda = params[..., :func.dim_N]

    def integrate(tt, ll):
        lm = (ll[:-1, ...] + ll[1:, ...]) / 2.0
        dts = (tt[1:] - tt[:-1]).reshape((-1,)+(1,)*(len(lm.shape)-1)).float()
        return (lm * dts).sum()

    log_likelihood = -integrate(tsave, lmbda)

    if func.evnt_embedding == "discrete":
        et_error = []
        for evnt in tsne:
            log_likelihood += torch.log(lmbda[evnt])
            type_preds = torch.zeros(len(type_forecast))
            for tid, t in enumerate(type_forecast):
                loc = (np.searchsorted(tsavenp, tsave[evnt[0]].item()-t),) + evnt[1:-1]
                type_preds[tid] = lmbda[loc].argmax().item()
            et_error.append((type_preds != evnt[-1]).float())

    elif func.evnt_embedding == "continuous":
        gsmean = params[..., func.dim_N*(1+func.dim_E*0):func.dim_N*(1+func.dim_E*1)].view(params.shape[:-1]+(func.dim_N, func.dim_E))
        logvar = params[..., func.dim_N*(1+func.dim_E*1):func.dim_N*(1+func.dim_E*2)].view(params.shape[:-1]+(func.dim_N, func.dim_E))
        var = torch.exp(logvar)

        def log_normal_pdf(loc, k):
            const = torch.log(torch.tensor(2.0*np.pi))
            return -0.5*(const + logvar[loc] + (gsmean[loc] - func.evnt_embed(k))**2.0 / var[loc])

        et_error = []
        for evnt in tsne:
            gsdensity = torch.exp(log_normal_pdf(evnt[:-1], evnt[-1]).sum(dim=-1)) + 1.0e-32  # add 1.0e-32, make sure the result is not zero
            tmp = torch.log((lmbda[evnt[:-1]] * gsdensity).sum(dim=-1))
            log_likelihood += tmp
            # mean_pred embedding
            mean_preds = torch.zeros(len(type_forecast), func.dim_E)
            for tid, t in enumerate(type_forecast):
                loc = (np.searchsorted(tsavenp, tsave[evnt[0]].item()-t),) + evnt[1:-1]
                mean_preds[tid] = ((lmbda[loc].view(func.dim_N, 1) * gsmean[loc]).sum(dim=0) / lmbda[loc].sum()).item()
            et_error.append((mean_preds - func.evnt_embed(evnt[-1])).norm(dim=-1)**2.0)

    METE = sum(et_error)/len(et_error)

    return tsave, trace, lmbda, gtid, tsne, -log_likelihood, METE


def poisson_lmbda(tmin, tmax, dt, lmbda0, TS):
    grid = np.round(np.arange(tmin, tmax+dt, dt), decimals=8)

    lmbda = []
    for _ in TS:
        lmbda.append(lmbda0 * np.ones(grid.shape))

    return lmbda


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
    if evnt_align:
        kernel_grid = np.arange(dt, 10.0 + 10.0*sigma, dt)
        kernel = np.concatenate(([0], alpha * (beta/sigma) * (kernel_grid / sigma)**(-beta-1) * (kernel_grid > sigma)))
        for ts in TS:
            vv = np.zeros(grid.shape)
            for record in ts:
                vv[t2tid[cl(record[0])]] = 1.0
            lmbda.append(lmbda0 + np.convolve(kernel, vv)[:grid.shape[0]])
    else:
        for ts in TS:
            vv = np.zeros(grid.shape)
            for record in ts:
                lo = t2tid[cl(min(record[0]+sigma, 100.0))]
                hi = t2tid[cl(min(record[0]+10.0+10.0*sigma, 100.0))]
                vv[lo:hi] += alpha * (beta/sigma) * ((grid[lo:hi]-record[0]) / sigma)**(-beta-1)
            lmbda.append(lmbda0 + vv)

    return lmbda


def self_inhibiting_lmbda(tmin, tmax, dt, mu, beta, TS, evnt_align=False):
    if evnt_align:
        tc = lambda t: np.round(np.ceil((t-tmin) / dt) * dt + tmin, decimals=8)
    else:
        tc = lambda t: t

    grid = np.round(np.arange(tmin, tmax+dt, dt), decimals=8)

    lmbda = []
    for ts in TS:
        log_lmbda = mu * (grid-tmin)
        for record in ts:
            log_lmbda[grid > tc(record[0])] -= beta
        lmbda.append(np.exp(log_lmbda))

    return lmbda


def read_timeseries(filename, num_seqs=sys.maxsize):
    with open(filename) as f:
        seqs = f.readlines()[:num_seqs]
    return [[(float(t), 0) for t in seq.split(';')[0].split()] for seq in seqs]
