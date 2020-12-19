#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estimate the cost of enumeration using Pruner/Simulator

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
from fpylll import BKZ, FPLLL, GSO, IntegerMatrix, LLL, Pruning
from fpylll.fplll.bkz_param import Strategy, dump_strategies_json
from fpylll.tools.bkz_simulator import simulate as bkz_simulate_fplll
from fpylll.util import gaussian_heuristic


def sample_matrix(d, lattice_type="qary", seed=None):
    """
    Sample a matrix in dimension `d`.

    :param d: lattice dimension
    :param lattice_type: see module level documentation
    :param seed: optional random seed
    :returns: LLL-reduced integer matrix

    .. note :: This function seeds the FPLLL RNG, i.e. it is deterministic.

    """

    if seed is None:
        FPLLL.set_random_seed(d)
    else:
        FPLLL.set_random_seed(seed)

    if lattice_type == "qary":
        A = IntegerMatrix.random(d, "qary", bits=30, k=d // 2, int_type="long")
    elif lattice_type == "qary-lv":
        A = IntegerMatrix.random(d, "qary", bits=10 * d, k=d // 2)
    else:
        raise ValueError("Lattice type '%s' not supported." % lattice_type)

    A = LLL.reduction(A)
    return A


def sample_r(d, lattice_type="qary", run_lll=False):
    """
    Sample squared Gram-Schmidt norms of an LLL reduced lattice in dimension d.

    :param d: lattice dimension
    :param lattice_type: see module level documentation
    :param run_lll: if ``True`` sample a matrix and run LLL

    """
    if run_lll:
        A = sample_matrix(d=d, lattice_type=lattice_type)

        if d < 160:
            M = GSO.Mat(A, float_type="d")
        elif d < 320:
            M = GSO.Mat(A, float_type="dd")
        else:
            M = GSO.Mat(A, float_type="qd")
        M.update_gso()

        return M.r()
    else:
        if lattice_type == "qary":
            q = 2 ** 30
        elif lattice_type == "qary-lv":
            q = 2 ** (10 * d)
        else:
            raise ValueError("Lattice type '%s' not supported." % lattice_type)
        return [1.0219 ** (2 * (d - 2 * i - 1)) * q for i in range(d)]


def lll_cost(d):
    """
    Cost of LLL in dimension `d` in enumeration nodes.

    :param d: lattice dimension

    .. note:: We are ignoring the bit-size of the input here.

    """
    return d ** 3


def _pruner_precision(d, greedy):
    """
    Required precision for pruner in dimension `d`.
    """
    if d < 126:
        return "d"
    elif d < 150:
        return "ld"
    elif d < 248:
        return "dd"
    elif d < 342:
        return "qd"
    else:
        return round(0.75 * d)


def _block_sizes(block_size, strategies):
    """
    Return list of block sizes run for a given enumeration.

    :param block_size: target, top-level block size
    :param strategies: preprocessing strategies

    EXAMPLE::

        >>> from fpylll import load_strategies_json, BKZ
        >>> strategies = load_strategies_json(BKZ.DEFAULT_STRATEGY)
        >>> from cost import _block_sizes
        >>> _block_sizes(80, strategies)
        [80, 58, 40]

    """
    block_size_list = []

    for block_size in strategies[block_size].preprocessing_block_sizes:
        block_size_list.extend(_block_sizes(block_size, strategies))
        block_size_list.append(block_size)

    return block_size_list


def preprocess(
    r,
    preproc_block_size,
    strategies,
    costs,
    max_loops=1,
    bkz_simulate=bkz_simulate_fplll,
):
    """
    Simulate preprocessing algorithm.

    :param r: squared Gram-Schmidt norms
    :param block_size: preprocessing block size
    :param strategies: strategies computed so far to establish recursive preprocessing
    :param costs: enumeration cost per dimension
    :param max_loops: number of loops to run during preprocessing (FPyLLL uses 1)
    :returns: new basis shape and preprocessing cost

    """
    d = len(r)

    for _ in range(max_loops):
        for preproc_block_size_ in _block_sizes(preproc_block_size, strategies):
            r = bkz_simulate(
                r,
                BKZ.EasyParam(
                    preproc_block_size_, strategies=strategies, max_loops=1
                ),
            )[0]
        r = bkz_simulate(
            r,
            BKZ.EasyParam(
                preproc_block_size, strategies=strategies, max_loops=1
            ),
        )[0]

    # we first run LLL
    cost = lll_cost(d)

    # And then roughly d x preprocessing, recursive calls are included in total cost
    for kappa in range(d - 3):
        cost += (
            max_loops * costs[min(d - kappa, preproc_block_size)]["total cost"]
        )
    return r, cost


def pruning_coefficients(
    r,
    preprocessing_cost,
    radius_bound=1,
    target_success_probability=0.51,
    float_type="d",
    greedy=False,
):
    """
    Compute pruning coefficients for r.

    :param r: squared Gram-Schmidt norms
    :param preprocessing_cost: cost of achieving basis of quality similar to r
    :param radius_bound: compute pruning parameters for `GH^(i/radius_bound)` for `i in -radius_bound, …, radius_bound`
    :param target_success_probability: target success probability of enumeration
    :param float_type: floating point type to use
    :param greedy: use Greedy pruning strategy
    :returns: tuple of ``Pruning`` objects.

    """
    d = len(r)
    gh = gaussian_heuristic(r)
    r = [r_ / gh for r_ in r]
    gh_margin = min(2.0, (1.0 + 3.0 / d)) ** 2

    pruning_coefficients_ = []
    for i in range(-radius_bound, radius_bound + 1):
        pruning = Pruning.run(
            gh_margin ** (1.0 * i / radius_bound),
            preprocessing_cost,
            [r],
            target_success_probability,
            float_type=float_type,
            flags=Pruning.HALF if greedy else Pruning.GRADIENT | Pruning.HALF,
        )
        pruning_coefficients_.append(pruning)
    return tuple(pruning_coefficients_)


def enumeration_cost(
    r,
    preproc,
    strategies,
    costs,
    gh_factor=1.00,
    preproc_loops=1,
    target_success_probability=0.51,
    float_type="d",
    greedy=False,
):
    """
    Cost of enumeration on `r` using ``strategies``.

    :param r: squared Gram-Schmidt vectors
    :param strategies: prepcomputed strategies
    :param costs: precomputed costs for smaller dimensions
    :param gh_factor: target GH_FACTOR * GH
    :param max_loops: number of preprocessing loops
    :param float_type: float type to use in pruner
    :param greedy: use Greedy pruning strategy.

    """
    d = len(r)

    if preproc is None:
        preproc_cost = lll_cost(d)
    else:
        r, preproc_cost = preprocess(
            r, preproc, strategies, costs, max_loops=preproc_loops
        )

    gh = gaussian_heuristic(r)

    target_norm = min(gh_factor * gh, r[0])

    pc = pruning_coefficients(
        r, preproc_cost, float_type=float_type, greedy=greedy
    )
    strategy = Strategy(
        d,
        preprocessing_block_sizes=[preproc] * preproc_loops
        if preproc > 2
        else [],
        pruning_parameters=pc,
    )
    pr = strategy.get_pruning(target_norm, gh)

    pruner = Pruning.Pruner(
        target_norm,
        preproc_cost,
        [r],
        target=target_success_probability,
        float_type=float_type,
        flags=Pruning.HALF if greedy else Pruning.GRADIENT | Pruning.HALF,
    )

    cost = {
        "total cost": preproc_cost + pruner.repeated_enum_cost(pr.coefficients),
        "single enum": pruner.single_enum_cost(pr.coefficients),
        "preprocessing": preproc_cost,
        "probability": pruner.measure_metric(pr.coefficients),
    }
    return cost, strategy


def cost_kernel(arg0, preproc=None, strategies=None, costs=None, opts=None):
    """
    Compute pruning coefficients after preprocessing and return estimated cost.

    :param arg0: either a tuple containing all arguments or r (squared Gram-Schmidt vectors)
    :param preproc: preprocessing parameters
    :param strategies: reduction strategies
    :param costs: precomputed costs for smaller dimensions
    :param opts: passed through to `enumeration_cost`

    :returns: cost and strategy

    ..  note :: the unusual arrangement with ``arg0`` is to support ``Pool.map`` which only
        supports one input parameter.
    """

    if (
        preproc is None
        and strategies is None
        and costs is None
        and opts is None
    ):
        r, preproc, strategies, costs, opts = arg0
    else:
        r = arg0

    float_type = opts["float_type"]

    if isinstance(float_type, int):
        FPLLL.set_precision(float_type)
        opts["float_type"] = "mpfr"

    try:
        cost, strategy = enumeration_cost(r, preproc, strategies, costs, **opts)
        return cost, strategy
    except RuntimeError:
        return None, None


def _prepare_parameters(
    dump_filename, strategies_and_costs, lower_limit, lattice_type, greedy=False
):
    if dump_filename is None:
        dump_filename = "../data/fplll-simulations,{lattice_type}{g}.sobj".format(
            lattice_type=lattice_type, g=",g" if greedy else ""
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


@begin.subcommand  # noqa
@begin.convert(
    upper_limit=int, lower_limit=int, ncores=int, greedy=bool, preproc_loops=int
)  # noqa
def strategize(
    upper_limit: "compute up to this dimension (inclusive)",
    lower_limit: """compute starting at this dimension,
               if ``None`` lowest unknown dimension is chosen.""" = None,
    strategies_and_costs: "previously computed strategies and costs to extend" = None,
    lattice_type: "one of 'qary' or 'qary-lv'" = "qary",
    dump_filename: """results are regularly written to this filename, if ``None``
               then ``../data/fplll-estimates-{lattice_type}.sobj`` is used.""" = None,
    ncores: "number of cores to use in parallel" = 4,
    greedy: "use Greedy pruning strategy" = False,
    preproc_loops: "number of preprocessing tours" = 2,
):
    """Estimate cost of enumeration.
    """

    dump_filename, strategies, costs, lower_limit = _prepare_parameters(
        dump_filename,
        strategies_and_costs,
        lower_limit,
        lattice_type,
        greedy=greedy,
    )

    if ncores > 1:
        workers = Pool(ncores)

    for d in range(lower_limit, upper_limit + 1):
        r = sample_r(d, lattice_type=lattice_type)

        float_type = _pruner_precision(d, greedy)

        try:
            start = strategies[d - 1].preprocessing_block_sizes[-1]
        except IndexError:
            start = 2

        if d < 60:
            stop = d
        else:
            stop = start + max(8, ncores)

        best = None

        for giant_step in range(start, stop, ncores):
            jobs, results = [], []
            for baby_step in range(giant_step, min(stop, giant_step + ncores)):
                opts = {
                    "greedy": greedy,
                    "float_type": float_type,
                    "preproc_loops": preproc_loops,
                }
                jobs.append((r, baby_step, strategies, costs, opts))

            if ncores == 1:
                for job in jobs:
                    results.append(cost_kernel(job))
            else:
                results = workers.map(cost_kernel, jobs)

            do_break = False
            for cost, strategy in results:
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
            "%3d :: %5.1f %s"
            % (d, log(costs[-1]["total cost"], 2), strategies[-1])
        )
        pickle.dump((strategies, costs), open(dump_filename, "wb"))
        dump_strategies_json(
            dump_filename.replace(".sobj", "-strategies.json"), strategies
        )

    return strategies, costs


@begin.subcommand  # noqa
@begin.convert(upper_limit=int, lower_limit=int, ncores=int, greedy=bool)
def extend(
    upper_limit: "compute up to this dimension (inclusive)",
    strategies_and_costs: "extend previously computed data",
    lower_limit: """compute starting at this dimension,
           if ``None`` lowest unknown dimension is chosen.""" = None,
    preprocessing: "function for selecting preprocessing block size" = "0.8605 * d - 14.04",
    lattice_type: "one of 'qary', 'qary-lv' or 'block'" = "qary",
    dump_filename: """results are regularly written to this filename, if ``None``
           then ``../data/fplll-estimates-{lattice_type}.sobj`` is used.""" = None,
    ncores: "number of cores to use in parallel" = 4,
    greedy: "use Greedy pruning strategy" = False,
):
    """Estimate cost of enumeration for fixed preprocessing block size as a function of the dimension.
    """

    dump_filename, strategies, costs, lower_limit = _prepare_parameters(
        dump_filename,
        strategies_and_costs,
        lower_limit,
        lattice_type,
        greedy=greedy,
    )

    preprocessing = eval("lambda d: round({})".format(preprocessing))

    if ncores > 1:
        workers = Pool(ncores)

    for step in range(lower_limit, upper_limit + 1, ncores):
        jobs, results = [], []
        for i, d in enumerate(range(step, min(step + ncores, upper_limit + 1))):
            float_type = _pruner_precision(d, greedy)
            r = sample_r(d, lattice_type=lattice_type)
            preproc = preprocessing(d)
            jobs.append(
                (
                    r,
                    preproc - 1,
                    strategies + [None] * i,
                    costs,
                    float_type,
                    greedy,
                )
            )
            jobs.append(
                (
                    r,
                    preproc + 0,
                    strategies + [None] * i,
                    costs,
                    float_type,
                    greedy,
                )
            )
            jobs.append(
                (
                    r,
                    preproc + 1,
                    strategies + [None] * i,
                    costs,
                    float_type,
                    greedy,
                )
            )

        if ncores == 1:
            for job in jobs:
                results.append(cost_kernel(job))
        else:
            results = workers.map(cost_kernel, jobs)

        for cost, strategy in results:
            try:
                if (
                    costs[strategy.block_size]["total cost"]
                    > cost["total cost"]
                ):
                    strategies[strategy.block_size] = strategy
                    costs[strategy.block_size] = cost
            except IndexError:
                strategies.append(strategy)
                costs.append(cost)

        for d in range(len(jobs) // 3)[::-1]:
            print(
                "%3d :: %5.1f %s"
                % (
                    strategies[-d - 1].block_size,
                    log(costs[-d - 1]["total cost"], 2),
                    strategies[-d - 1],
                )
            )

        pickle.dump((strategies, costs), open(dump_filename, "wb"))
        dump_strategies_json(
            dump_filename.replace(".sobj", "-strategies.json"), strategies
        )

    return strategies, costs


@begin.start
@begin.logging
def main():
    pass
