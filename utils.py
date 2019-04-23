import os
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
        for nid in range(lmbda.shape[2]):
            fig = plt.figure(figsize=(6, 6), facecolor='white')
            axe = plt.gca()
            axe.set_title('Point Process Modeling')
            axe.set_xlabel('time')
            axe.set_ylabel('intensity')
            axe.set_ylim(-10.0, 10.0)

            # plot the state function
            if (tsave is not None) and (trace is not None):
                for dat in list(trace[:, sid, nid, :].detach().numpy().T):
                    plt.plot(tsave.numpy(), dat, linewidth=0.3)

            # plot the state function (backward trace)
            if (tsave_ is not None) and (trace_ is not None):
                for dat in list(trace_[:, sid, nid, :].detach().numpy().T):
                    plt.plot(tsave_.numpy(), dat, linewidth=0.2, linestyle="dotted", color="black")

            # plot the intensity function
            if (grid is not None) and (lmbda_real is not None):
                plt.plot(grid.numpy(), lmbda_real[sid], linewidth=1.0, color="gray")
            plt.plot(tsave.numpy(), lmbda[:, sid, nid, :].detach().numpy(), linewidth=0.7)

            if tsne is not None:
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


def forward_pass(func, z0, tspan, dt, batch, evnt_align):
    # merge the sequences to create a sequence
    # t(ime)s(equence)n(ode)e(vent)
    batch_record = sorted([(record[0],) + (sid,) + record[1:]
                           for sid in range(len(batch)) for record in batch[sid]])

    # set up grid
    tsave, gtid, evnt_record, tsne = create_tsave(tspan[0], tspan[1], dt, batch_record, evnt_align)
    func.set_evnts(evnts=evnt_record)

    # forward pass
    trace = odeint(func, z0.repeat(len(batch), 1, 1), tsave, method='jump_adams', rtol=1.0e-5, atol=1.0e-7)
    lmbda = func.L(trace)

    def integrate(tt, ll):
        dts = tt[1:] - tt[:-1]
        return ((ll[:-1, :, :, :] + ll[1:, :, :, :]) / 2.0 * dts.reshape(-1, 1, 1, 1).float()).sum()

    loss = -(sum([torch.log(lmbda[record]) for record in tsne]) - integrate(tsave, lmbda))

    return tsave, trace, lmbda, gtid, tsne, loss


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
