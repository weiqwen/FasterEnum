#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tools for establishing the quality of (Procrastinating)-BKZ
"""

import begin
from collections import OrderedDict

from fpylll import FPLLL, IntegerMatrix, GSO, LLL, BKZ
from fpylll.tools.quality import get_current_slope
from math import exp, ceil
import csv
from multiprocessing import Pool


@begin.subcommand
@begin.convert(
    upper_limit=int,
    lower_limit=int,
    step_size=int,
    dfact=float,
    max_loops=int,
    c=float,
    sd=bool,
)
def rhf(
    upper_limit: "Compute up to this dimension (inclusive).",
    lower_limit: "Compute starting at this dimension, if ``None`` dimension 2 is chosen." = 2,
    step_size: "Increase dimension by this much in each step." = 1,
    dfact: "Dimension of the lattice is larger by this factor than block size" = 2,
    max_loops: "Maximum number of BKZ tours." = 8,
    c: "Overshoot parameter." = 0.25,
    sd: "Use self-dual ProcrastinatingBKZ." = False,
    plain_strat: "Strategies used for BKZ" = "../data/fplll-simulations,qary-strategies.json",
    spice_strat: "Strategies used for Procrastinating BKZ" = "../data/fplll-block-simulations,qary,0.25-strategies.json",
    filename: "Output filename" = "../data/bkz-rhf-{dfact:.2f},{c:.2f}{sd}.csv",
):
    """
    Print and save δ for different BKZ variants.
    """
    from cost import sample_r
    from simu import BKZQualitySimulation, ProcrastinatingBKZQualitySimulation

    PLAIN, SPICE = OrderedDict(), OrderedDict()

    filename = filename.format(c=c, dfact=dfact, sd=",sd" if sd else "")

    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        csvwriter.writerow(
            ["beta", "bkzrho", "bkzdelta", "procrastbkzrho", "procrastbkzdelta"]
        )
        for beta in range(lower_limit, upper_limit + step_size, step_size):
            r = sample_r(round(dfact * beta))
            params = BKZ.EasyParam(
                block_size=beta, max_loops=max_loops, strategies=plain_strat
            )
            plain = BKZQualitySimulation(list(r))(params)

            if sd:
                params = BKZ.EasyParam(
                    block_size=beta,
                    max_loops=max_loops // 2,
                    c=c,
                    strategies=spice_strat,
                )
                spice = [1.0 / r_ for r_ in reversed(r)]
                spice = ProcrastinatingBKZQualitySimulation(list(spice))(params)
                spice = [1.0 / r_ for r_ in reversed(spice)]
                spice = ProcrastinatingBKZQualitySimulation(list(spice))(params)
            else:
                params = BKZ.EasyParam(
                    block_size=beta,
                    max_loops=max_loops,
                    strategies=spice_strat,
                    c=c,
                )
                spice = ProcrastinatingBKZQualitySimulation(list(r))(params)

            plain_rho = get_current_slope(plain, 0, len(r))
            spice_rho = get_current_slope(spice, 0, len(r))
            plain_dlt = exp(-plain_rho / 4)
            spice_dlt = exp(-spice_rho / 4)

            PLAIN[beta] = plain_dlt
            SPICE[beta] = spice_dlt

            fmt = "β: {beta:3d}, δ_l: {plain_dlt:10.7f}, δ_r: {spice_dlt:10.7f}, δ_r/δ_l: {div:10.7f}"
            print(
                fmt.format(
                    beta=beta,
                    plain_dlt=plain_dlt,
                    spice_dlt=spice_dlt,
                    div=spice_dlt / plain_dlt,
                )
            )
            csvwriter.writerow(
                [beta, plain_rho, plain_dlt, spice_rho, spice_dlt]
            )

    return PLAIN, SPICE


def gso_workerf(args):
    import copy

    d, q, seed, params, procrastinating, what = args

    dummy = [1.0] * d

    if procrastinating:
        from impl import BKZReduction
        from simu import ProcrastinatingBKZSimulation as BKZSimulation
        from simu import (
            ProcrastinatingBKZQualitySimulation as BKZQualitySimulation,
        )
    else:
        from fpylll.algorithms.bkz2 import BKZReduction
        from simu import BKZSimulation
        from simu import BKZQualitySimulation

    FPLLL.set_random_seed(seed)
    A = LLL.reduction(IntegerMatrix.random(d, "qary", k=d // 2, q=q))

    if "qs" in what:
        qsimu_r = BKZQualitySimulation(copy.copy(A))(params)
    else:
        qsimu_r = dummy

    if "fs" in what:
        fsimu_r = BKZSimulation(copy.copy(A))(params)
    else:
        fsimu_r = dummy

    if "r" in what:
        BKZReduction(A)(params)
        M = GSO.Mat(A)
        M.update_gso()
        real_r = M.r()
    else:
        real_r = dummy

    return qsimu_r, fsimu_r, real_r


@begin.subcommand
@begin.convert(
    beta=int,
    max_loops=int,
    c=float,
    trials=int,
    workers=int,
    procrastinating=bool,
    what=str,
    d=int,
)
def gso(
    beta: "Block size.",
    strategies: "Strategies." = "../data/fplll-block-simulations,qary,0.25-strategies.json",
    max_loops: "Maximum number of BKZ tours." = 2,
    c: "Overshoot parameter, only used in Procrastinating-BKZ." = 0.25,
    d: "Dimension (default (1+c)β)." = None,
    trials: "Number of trials to run." = 8,
    workers: "Number of workers to use." = 1,
    procrastinating: "Consider Procrastinating-BKZ." = True,
    what: "A string containing, 'qs', 'fs' and/or 'r'" = "qs,fs,r",
    filename: "Output filename" = "../data/{procrastinating}bkz-gso-{beta:d},{d:d},{max_loops:d},{c:.2f},{what}.csv",
):

    if not d:
        d = ceil((1 + c) * beta)

    q = 1073741789
    challenges = []

    what = what.split(",")

    params = BKZ.EasyParam(
        block_size=beta,
        strategies=strategies,
        max_loops=max_loops,
        c=c,
        flags=BKZ.VERBOSE,
    )

    for i in range(trials):
        challenges.append((d, q, 0x1337 + i, params, procrastinating, what))

    if workers > 1:
        p = Pool(workers)
        res = list(p.imap(gso_workerf, challenges))
    else:
        res = list(map(gso_workerf, challenges))

    qs_r, fs_r, r_r = [], [], []
    for i in range(trials):
        qs_r.append(res[i][0])
        fs_r.append(res[i][1])
        r_r.append(res[i][2])

    filename = filename.format(
        beta=beta,
        max_loops=max_loops,
        c=c,
        d=d,
        procrastinating="procrastinating-" if procrastinating else "",
        what=",".join(what),
    )

    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        csvwriter.writerow(["i"] + list(map(lambda x: x + "_bi*", what)))
        for i in range(d):
            row = [i]
            if "qs" in what:
                row.append(sum(r_[i] for r_ in qs_r) / trials)
            if "fs" in what:
                row.append(sum(r_[i] for r_ in fs_r) / trials)
            if "r" in what:
                row.append(sum(r_[i] for r_ in r_r) / trials)
            csvwriter.writerow(row)

    return


@begin.start
@begin.logging
def run():
    pass
