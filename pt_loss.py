import sys
import torch
from utils import poisson_lmbda, exponential_hawkes_lmbda, powerlaw_hawkes_lmbda, self_inhibiting_lmbda

def read_timeseries(filename, num_seqs=sys.maxsize):
    with open(filename) as f:
        seqs = f.readlines()[:num_seqs]
    return [[(float(t), vid, 0) for vid, vts in enumerate(seq.split(";")) for t in vts.split()] for seq in seqs]


if __name__ == '__main__':
    dt, tspan = 0.05, (0.0, 100.0)
    TS_poisson = read_timeseries("data/point_processes/poisson_testing.csv")
    TS_exponential_hawkes = read_timeseries("data/point_processes/exponential_hawkes_testing.csv")
    TS_powerlaw_hawkes = read_timeseries("data/point_processes/powerlaw_hawkes_testing.csv")
    TS_self_inhibiting = read_timeseries("data/point_processes/self_inhibiting_testing.csv")

    real_m1 = poisson_lmbda(tspan[0], tspan[1], dt, 0.2, TS_poisson)
    real_m2 = exponential_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 1.0, TS_exponential_hawkes, False)
    real_m3 = powerlaw_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2, 0.8, 2.0, 1.0, TS_powerlaw_hawkes, False)
    real_m4 = self_inhibiting_lmbda(tspan[0], tspan[1], dt, 0.5, 0.2, TS_self_inhibiting, False)

    fit_m1_m1 = poisson_lmbda(tspan[0], tspan[1], dt, 0.2072, TS_poisson)
    fit_m1_m2 = exponential_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2072, 0.0, 1.0, TS_exponential_hawkes, False)
    fit_m1_m3 = powerlaw_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2066, 0.0038, 2.0, 1.0, TS_powerlaw_hawkes, False)
    fit_m1_m4 = self_inhibiting_lmbda(tspan[0], tspan[1], dt, 5.932e-11, 0.1333, TS_self_inhibiting, False)

    fit_m2_m1 = poisson_lmbda(tspan[0], tspan[1], dt, 1.003, TS_poisson)
    fit_m2_m2 = exponential_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2115, 0.8084, 1.0, TS_exponential_hawkes, False)
    fit_m2_m3 = powerlaw_hawkes_lmbda(tspan[0], tspan[1], dt, 0.3845, 0.6404, 2.0, 1.0, TS_powerlaw_hawkes, False)
    fit_m2_m4 = self_inhibiting_lmbda(tspan[0], tspan[1], dt, 0.0274, 0.5816, TS_self_inhibiting, False)

    fit_m3_m1 = poisson_lmbda(tspan[0], tspan[1], dt, 0.550, TS_poisson)
    fit_m3_m2 = exponential_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2930, 0.4815, 1.0, TS_exponential_hawkes, False)
    fit_m3_m3 = powerlaw_hawkes_lmbda(tspan[0], tspan[1], dt, 0.2214, 0.6234, 2.0, 1.0, TS_powerlaw_hawkes, False)
    fit_m3_m4 = self_inhibiting_lmbda(tspan[0], tspan[1], dt, 2.490e-11, 0.011, TS_self_inhibiting, False)

    fit_m4_m1 = poisson_lmbda(tspan[0], tspan[1], dt, 2.466, TS_poisson)
    fit_m4_m2 = exponential_hawkes_lmbda(tspan[0], tspan[1], dt, 2.466, 0.0, 1.0, TS_exponential_hawkes, False)
    fit_m4_m3 = powerlaw_hawkes_lmbda(tspan[0], tspan[1], dt, 2.466, 0.0, 2.0, 1.0, TS_powerlaw_hawkes, False)
    fit_m4_m4 = self_inhibiting_lmbda(tspan[0], tspan[1], dt, 0.4979, 0.199, TS_self_inhibiting, False)

    lmbda_loss = lambda lmbda1, lmbda2: ((torch.tensor(lmbda1).float() - torch.tensor(lmbda2).float()).abs()*dt/100.05/100).sum()
    lmbda_ratio_loss = lambda lmbda1, lmbda2: (((torch.tensor(lmbda1).float()-torch.tensor(lmbda2).float())/torch.tensor(lmbda2).float()).abs()*dt/100.05/100).sum()

    loss = [[lmbda_loss(fit_m1_m1, real_m1), lmbda_loss(fit_m2_m1, real_m2), lmbda_loss(fit_m3_m1, real_m3), lmbda_loss(fit_m4_m1, real_m4)], 
            [lmbda_loss(fit_m1_m2, real_m1), lmbda_loss(fit_m2_m2, real_m2), lmbda_loss(fit_m3_m2, real_m3), lmbda_loss(fit_m4_m2, real_m4)],
            [lmbda_loss(fit_m1_m3, real_m1), lmbda_loss(fit_m2_m3, real_m2), lmbda_loss(fit_m3_m3, real_m3), lmbda_loss(fit_m4_m3, real_m4)],
            [lmbda_loss(fit_m1_m4, real_m1), lmbda_loss(fit_m2_m4, real_m2), lmbda_loss(fit_m3_m4, real_m3), lmbda_loss(fit_m4_m4, real_m4)]]

    ratio_loss = [[lmbda_ratio_loss(fit_m1_m1, real_m1), lmbda_ratio_loss(fit_m2_m1, real_m2), lmbda_ratio_loss(fit_m3_m1, real_m3), lmbda_ratio_loss(fit_m4_m1, real_m4)],
                  [lmbda_ratio_loss(fit_m1_m2, real_m1), lmbda_ratio_loss(fit_m2_m2, real_m2), lmbda_ratio_loss(fit_m3_m2, real_m3), lmbda_ratio_loss(fit_m4_m2, real_m4)],
                  [lmbda_ratio_loss(fit_m1_m3, real_m1), lmbda_ratio_loss(fit_m2_m3, real_m2), lmbda_ratio_loss(fit_m3_m3, real_m3), lmbda_ratio_loss(fit_m4_m3, real_m4)],
                  [lmbda_ratio_loss(fit_m1_m4, real_m1), lmbda_ratio_loss(fit_m2_m4, real_m2), lmbda_ratio_loss(fit_m3_m4, real_m3), lmbda_ratio_loss(fit_m4_m4, real_m4)]]
