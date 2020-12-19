#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estimate the cost of enumeration using Pruner/Simulator in a BKZ block.

That, is we assume we can push the HKZ part of the shape beyond the enumeration context.

Supported integer matrix types
------------------------------

- qary :: q-ary lattice with `q ≈ 2^{30}`
- qary-lv :: q-ary lattice with `q ≈ 2^{10⋅d}`


"""

import logging
import pickle
from math import log
from multiprocessing import Pool

import begin
from fpylll import Pruning, FPLLL
from fpylll.fplll.bkz_param import Strategy, dump_strategies_json
from fpylll.util import gaussian_heuristic
from simu import bkz_simulatef, ProcrastinatingBKZQualitySimulation, SDProcrastinatingBKZQualitySimulation
from math import ceil


def preprocess(r, c, preproc_block_size, strategies, costs, max_loops=1, bkz_simulate=None):
    """
    Simulate preprocessing algorithm.

    :param r: squared Gram-Schmidt norms
    :param block_size: preprocessing block size
    :param strategies: strategies computed so far to establish recursive preprocessing
    :param costs: enumeration cost per dimension
    :param max_loops: number of loops to run during preprocessing (FPyLLL uses 1)
    :returns: new basis shape and preprocessing cost

    """
    D = len(r)

    from cost import lll_cost, BKZ

    r = bkz_simulate(r, BKZ.EasyParam(preproc_block_size, strategies=strategies, max_loops=max_loops))[0]

    # we first run LLL
    cost = lll_cost(D)

    limit = ceil((1 + c) * preproc_block_size)

    for kappa in range(0, D - limit):
        # We run SVP reductions with cost β^{β/8 + o(β)}
        cost += max_loops * costs[preproc_block_size]["total cost"]

    cost_ceil = log(preproc_block_size) * preproc_block_size / 8.0

    for i, kappa in enumerate(range(D - limit, D - 3)):
        block_size = max(preproc_block_size - int(ceil((i + 1) / 2)), 2)
        while cost_ceil > 0.184 * log(block_size + 1) * (block_size + 1):
            block_size += 1
        block_size = min(D - kappa, block_size)
        cost += max_loops * costs[block_size]["total cost"]

    return r, cost


def enumeration_cost(
    r,
    d,
    c,
    preproc,
    strategies,
    costs,
    gh_factor=1.10,
    float_type="d",
    greedy=False,
    sd=False,
    radius_bound=1,
    preproc_loops=None,
    ignore_preproc_cost=False,
):
    """
    Cost of enumeration on `r` using ``strategies``.

    :param r: squared Gram-Schmidt vectors
    :param d: enumeration dimension
    :param c: overshoot parameter
    :param preproc: preprocessing dimension
    :param strategies: prepcomputed strategies
    :param costs: precomputed costs for smaller dimensions
    :param gh_factor: target GH_FACTOR * GH
    :param float_type: float type to use in pruner
    :param greedy: use Greedy pruning strategy.
    :param sd: use self-dual strategy
    :param radius_bound: compute pruning parameters for `GH^(i/radius_bound)` for `i in -radius_bound, …, radius_bound`
    :param preproc_loops: number of loops to perform preprocessing for
    :param ignore_preproc_cost: assume all preprocessing has the cost of LLL regardless of block size.

    """
    from cost import lll_cost, pruning_coefficients

    D = int((1 + c) * d)

    if preproc is None or preproc == 2:
        preproc_cost = lll_cost(D)
        r_ = list(r)
    else:
        if sd:
            f = bkz_simulatef(
                SDProcrastinatingBKZQualitySimulation,
                init_kwds={"preprocessing_levels": 1, "preprocessing_cutoff": 45},
                call_kwds={"c": c},
            )
            r_, preproc_cost = preprocess(r[:D], c, preproc, strategies, costs, max_loops=preproc_loops, bkz_simulate=f)
            # each SD tour costs as much as one BKZ tour
            preproc_cost = 2 * preproc_cost
        else:
            f = bkz_simulatef(
                ProcrastinatingBKZQualitySimulation,
                init_kwds={"preprocessing_levels": 1, "preprocessing_cutoff": 45},
                call_kwds={"c": c},
            )
            r_, preproc_cost = preprocess(r[:D], c, preproc, strategies, costs, max_loops=preproc_loops, bkz_simulate=f)

    if ignore_preproc_cost:
        preproc_cost = lll_cost(D)

    gh = gaussian_heuristic(r_[:d])
    target_norm = gh_factor * gh

    pc = pruning_coefficients(r_[:d], preproc_cost, radius_bound=radius_bound, float_type=float_type, greedy=greedy)
    strategy = Strategy(
        d, preprocessing_block_sizes=[preproc] * preproc_loops if preproc > 2 else [], pruning_parameters=pc
    )
    pr = strategy.get_pruning(target_norm, gh)

    pruner = Pruning.Pruner(
        target_norm,
        preproc_cost,
        [r_[:d]],
        target=0.51,
        float_type=float_type,
        flags=Pruning.HALF if greedy else Pruning.GRADIENT | Pruning.HALF,
    )

    cost = {
        "total cost": preproc_cost + pruner.repeated_enum_cost(pr.coefficients),
        "single enum": pruner.single_enum_cost(pr.coefficients),
        "preprocessing": preproc_cost,
        "c": c,
        "probability": pruner.measure_metric(pr.coefficients),
    }

    logging.debug(
        "%3d :: C: %5.1f, P: %5.1f c: %.2f, %s"
        % (d, log(cost["total cost"], 2), log(cost["preprocessing"], 2), cost["c"], strategy)
    )

    return cost, strategy


def cost_kernel(arg0, d=None, c=None, preproc=None, strategies=None, costs=None, opts=None):
    """
    Compute pruning coefficients after preprocessing and return estimated cost.

    :param arg0: either a tuple containing all arguments or r (squared Gram-Schmidt vectors)
    :param d: enumeration dimension
    :param c: overshoot parameter
    :param preproc: preprocessing parameters
    :param strategies: reduction strategies
    :param costs: precomputed costs for smaller dimensions
    :param opts: passed through to `enumeration_cost`

    :returns: cost and strategy

    ..  note :: the unusual arrangement with ``arg0`` is to support ``Pool.map`` which only
        supports one input parameter.
    """

    if preproc is None and c is None and strategies is None and costs is None and opts is None:
        r, d, c, preproc, strategies, costs, opts = arg0
    else:
        r = arg0

    float_type = opts["float_type"]

    if isinstance(float_type, int):
        FPLLL.set_precision(float_type)
        opts["float_type"] = "mpfr"

    try:
        return enumeration_cost(r, d, c, preproc, strategies, costs, **opts)
    except RuntimeError:
        return None, None


def _prepare_parameters(
    dump_filename,
    c,
    strategies_and_costs,
    lower_limit,
    lattice_type,
    preproc_loops,
    greedy=False,
    sd=False,
    ignore_preproc_cost=False,
):
    if dump_filename is None:
        dump_filename = "../data/fplll-block-simulations,{lattice_type},{c:.2f},{preproc_loops:d}{g}{sd}{lb}.sobj".format(
            lattice_type=lattice_type,
            c=c,
            preproc_loops=preproc_loops,
            g=",g" if greedy else "",
            sd=",sd" if sd else "",
            lb=",lb" if ignore_preproc_cost else "",
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

    if lower_limit is None:
        lower_limit = len(strategies)
    else:
        strategies = strategies[:lower_limit]
        costs = costs[:lower_limit]

    return dump_filename, strategies, costs, lower_limit


@begin.start
@begin.logging
@begin.convert(
    upper_limit=int,
    lower_limit=int,
    ncores=int,
    c=float,
    greedy=bool,
    sd=bool,
    gh_factor=float,
    rb=int,
    preproc_loops=int,
    ignore_preproc_cost=bool,
)  # noqa
def block_strategize(
    upper_limit: "compute up to this dimension (inclusive)",
    lower_limit: """compute starting at this dimension,
        if ``None`` lowest unknown dimension is chosen.""" = None,
    c: "overshoot parameter" = 0.25,
    strategies_and_costs: "previously computed strategies and costs to extend" = None,
    lattice_type: "one of 'qary' or 'qary-lv'" = "qary",
    dump_filename: """results are regularly written to this filename, if ``None``
                     then ``data/fplll-block-simulations-{lattice_type}.sobj`` is used.""" = None,
    ncores: "number of cores to use in parallel" = 4,
    gh_factor: "set target_norm^2 to gh_factor * gh^2" = 1.00,
    rb: "compute pruning parameters for `GH^(i/rb)` for `i in -rb, …, rb`" = 1,
    greedy: "use Greedy pruning strategy" = False,
    sd: "use self-dual strategy" = False,
    preproc_loops: "number of preprocessing tours" = 2,
    ignore_preproc_cost: "assume all preprocessing has the cost of LLL regardless of block size" = False,
):
    """Estimate cost of enumeration.
    """

    dump_filename, strategies, costs, lower_limit = _prepare_parameters(
        dump_filename,
        c,
        strategies_and_costs,
        lower_limit,
        lattice_type,
        preproc_loops,
        greedy,
        sd,
        ignore_preproc_cost,
    )

    if ncores > 1:
        workers = Pool(ncores)

    from cost import sample_r, _pruner_precision

    for d in range(lower_limit, upper_limit + 1):
        D = int((1 + c) * d + 1)
        r = sample_r(D, lattice_type=lattice_type)

        float_type = _pruner_precision(d, greedy)

        try:
            start = max(strategies[d - 1].preprocessing_block_sizes[-1], 2)
        except IndexError:
            start = 2

        if d < 60:
            stop = d
        else:
            stop = min(start + max(8, ncores), d)

        best = None

        for giant_step in range(start, stop, ncores):
            jobs, results = [], []
            for baby_step in range(giant_step, min(stop, giant_step + ncores)):
                opts = {
                    "greedy": greedy,
                    "sd": sd,
                    "gh_factor": gh_factor,
                    "float_type": float_type,
                    "radius_bound": rb,
                    "preproc_loops": preproc_loops,
                    "ignore_preproc_cost": ignore_preproc_cost,
                }
                jobs.append((r, d, c, baby_step, strategies, costs, opts))

            if ncores == 1:
                for job in jobs:
                    results.append(cost_kernel(job))
            else:
                results = workers.map(cost_kernel, jobs)

            do_break = False
            for cost, strategy in results:
                logging.debug(
                    "%3d :: C: %5.1f, P: %5.1f c: %.2f, %s"
                    % (d, log(cost["total cost"], 2), log(cost["preprocessing"], 2), cost["c"], strategy)
                )
                if best is None or cost["total cost"] < best[0]["total cost"]:
                    best = cost, strategy
                if cost["total cost"] > 1.1 * best[0]["total cost"]:
                    do_break = True
                    break
            if do_break:
                break

        costs.append(best[0])
        strategies.append(best[1])
        logging.info(
            "%3d :: C: %5.1f, P: %5.1f c: %.2f, %s"
            % (d, log(costs[-1]["total cost"], 2), log(costs[-1]["preprocessing"], 2), costs[-1]["c"], strategies[-1])
        )
        pickle.dump((strategies, costs), open(dump_filename, "wb"))
        dump_strategies_json(dump_filename.replace(".sobj", "-strategies.json"), strategies)

    return strategies, costs
