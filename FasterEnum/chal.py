#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import OrderedDict
from fpylll import IntegerMatrix, GSO, Pruning, LLL
from fpylll.fplll.bkz_param import Strategy
from fpylll.fplll.pruner import svp_probability
from fpylll.util import gaussian_heuristic
from math import log
from multiprocessing import Pool
import begin
import logging
import pickle
import os


def cost_kernel(
    arg0, preproc=None, strategies=None, costs=None, float_type=None
):
    """
    Compute pruning coefficients after preprocessing and return estimated cost.

    :param arg0: either a tuple containing all arguments or r (squared Gram-Schmidt vectors)
    :param preproc: preprocessing parameters
    :param strategies: reduction strategies
    :param costs: precomputed costs for smaller dimensions
    :param float_type: float type to use in pruner

    :returns: cost and strategy

    ..  note :: the unusual arrangement with ``arg0`` is to support ``Pool.map`` which only
        supports one input parameter.
    """
    from cost import preprocess

    if (
        preproc is None
        and strategies is None
        and costs is None
        and float_type is None
    ):
        r, preproc, strategies, costs, float_type = arg0
    else:
        r = arg0

    d = len(r)

    r, preproc_cost = preprocess(r, preproc, strategies, costs, max_loops=1)

    gh = gaussian_heuristic(r)
    target_norm = 1.05 ** 2 * gh

    pruner = Pruning.Pruner(
        target_norm,
        preproc_cost,
        [r],
        target=1,
        metric=Pruning.EXPECTED_SOLUTIONS,
        float_type=float_type,
    )
    coefficients = pruner.optimize_coefficients([1.0] * d)
    cost = {
        "total cost": preproc_cost + pruner.repeated_enum_cost(coefficients),
        "single enum": pruner.single_enum_cost(coefficients),
        "preprocessing block size": preproc,
        "preprocessing": preproc_cost,
        "probability": svp_probability(coefficients, float_type=float_type),
    }
    return cost


def load_svp_challenge(d, seed=0):
    """
    Load SVP Challenge matrix in dimension `d` and ``seed``

    :param d: dimension
    :param seed: random seed

    """
    filename = os.path.join("data", "svp-challenge", "%03d-%d.txt" % (d, seed))

    if os.path.isfile(filename) is False:
        import requests

        logging.debug(
            "Did not find '{filename}', downloading ...".format(
                filename=filename
            )
        )
        r = requests.post(
            "https://www.latticechallenge.org/svp-challenge/generator.php",
            data={"dimension": d, "seed": seed, "sent": "True"},
        )
        logging.debug("%s %s" % (r.status_code, r.reason))
        fn = open(filename, "w")
        fn.write(r.text)
        fn.close()

    A = IntegerMatrix.from_file(filename)
    LLL.reduction(A)
    return A


def load_svp_challenge_r(d, seed=0):
    """
    Load SVP Challenge in dimension `d` and ``seed``

    :param d: dimension
    :param seed: random seed

    """
    A = load_svp_challenge(d, seed=seed)
    if d < 160:
        M = GSO.Mat(A, float_type="d")
    elif d < 320:
        M = GSO.Mat(A, float_type="dd")
    else:
        M = GSO.Mat(A, float_type="qd")
    M.update_gso()
    return M.r()


@begin.start  # noqa
@begin.logging
@begin.convert(upper_limit=int, lower_limit=int, ncores=int)  # noqa
def svp_challenge(
    upper_limit: "compute up to this dimension (inclusive)",
    strategies_and_costs: "previously computed strategies and costs to extend",
    lower_limit: """compute starting at this dimension,
                  if ``None`` lowest unknown dimension is chosen.""" = None,
    dump_filename: """results are regularly written to this filename, if ``None``
                  then ``data/fplll-estimates-{lattice_type}.sobj`` is used.""" = None,
    ncores: "number of cores to use in parallel" = 4,
):

    from cost import _pruner_precision

    if dump_filename is None:
        dump_filename = os.path.join(
            "data", "fplll-simulations,svp-challenge.sobj"
        )

    if strategies_and_costs is not None:
        try:
            strategies, costs = strategies_and_costs
        except ValueError:
            strategies, costs = pickle.load(open(strategies_and_costs, "rb"))
    else:
        costs, strategies = [], []
        for i in range(3):
            strategies.append(Strategy(i, [], []))
            costs.append({"total cost": 0.0})

    if ncores > 1:
        workers = Pool(ncores)

    scc = OrderedDict()

    for d in range(lower_limit, upper_limit + 1):
        try:
            r = load_svp_challenge_r(d, seed=0)
        except FileNotFoundError:
            continue

        float_type, precision = _pruner_precision(d)

        try:
            start = max(strategies[d].preprocessing_block_sizes[0] - 16, 2)
        except (KeyError, IndexError):
            start = 2

        stop = d
        best = None
        for giant_step in range(start, stop, ncores):
            jobs, results = [], []
            for baby_step in range(giant_step, min(stop, giant_step + ncores)):
                jobs.append((r, baby_step, strategies, costs, float_type))

            if ncores == 1:
                for job in jobs:
                    results.append(cost_kernel(job))
            else:
                results = workers.map(cost_kernel, jobs)

            do_break = False
            for cost in results:
                if best is None or cost["total cost"] < best["total cost"]:
                    best = cost
                if cost["total cost"] > 2 * best["total cost"]:
                    do_break = True
                    break
            if do_break:
                break

        scc[d] = best
        logging.info(
            "%3d :: %5.1f, %3d"
            % (d, log(best["total cost"], 2), best["preprocessing block size"])
        )
        pickle.dump(scc, open(dump_filename, "wb"))
