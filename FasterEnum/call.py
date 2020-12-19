#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run SVP, oSVP or HSVP_{1.05} reduction and record statistics.

"""
from collections import OrderedDict
from fpylll import FPLLL, BKZ, GSO, Pruning, Enumeration, EnumerationError, load_strategies_json
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from fpylll.tools.bkz_stats import BKZTreeTracer
from fpylll.util import gaussian_heuristic
from multiprocessing import Queue, Process, active_children
import begin
import logging
import pickle
import os
from math import log, ceil
from impl import BKZReduction as OBKZ

# Verbose logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(name)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S %Z')

# define a Handler which writes INFO messages or higher to the sys.stderr
logger = logging.getLogger(__name__)


def chunk_iterator(lst, step):
    """Return up to ``step`` entries from ``lst`` each time this function is called.

    :param lst: a list
    :param step: number of elements to return

    """
    for i in range(0, len(lst), step):
        yield tuple(lst[j] for j in range(i, min(i+step, len(lst))))


def svp_time(seed, params, return_queue=None):
    """Run SVP reduction on ``A`` using ``params``.

    :param seed: random seed for matrix creation
    :param params: BKZ parameters
    :param return_queue: if not ``None``, the result is put on this queue.

    """
    from cost import sample_matrix

    A = sample_matrix(params.block_size, seed=seed)
    M = GSO.Mat(A)
    bkz = BKZ2(M)
    tracer = BKZTreeTracer(bkz, start_clocks=True)

    with tracer.context(("tour", 0)):
        bkz.svp_reduction(0, params.block_size, params, tracer)
        bkz.M.update_gso()

    tracer.exit()

    tracer.trace.data["|A_0|"] = A[0].norm()
    ppbs = params.strategies[params.block_size].preprocessing_block_sizes
    tracer.trace.data["preprocessing_block_size"] = ppbs[0] if ppbs else 2

    if return_queue:
        return_queue.put(tracer.trace)
    else:
        return tracer.trace


def osvp_time(seed, params, return_queue=None):
    """Run oSVP reduction on ``A`` using ``params``.

    :param seed: random seed for matrix creation
    :param params: BKZ parameters
    :param return_queue: if not ``None``, the result is put on this queue.

    """
    from cost import sample_matrix

    A = sample_matrix(ceil(params.block_size *(1 + params["c"])), seed=seed)
    M = GSO.Mat(A)
    bkz = OBKZ(M, c=params["c"])
    tracer = BKZTreeTracer(bkz, start_clocks=True)

    with tracer.context(("tour", 0)):
        bkz.svp_reduction(0, params.block_size, params, tracer)
        bkz.M.update_gso()

    tracer.exit()

    tracer.trace.data["|A_0|"] = A[0].norm()
    ppbs = params.strategies[params.block_size].preprocessing_block_sizes
    tracer.trace.data["preprocessing_block_size"] = ppbs[0] if ppbs else 2

    if return_queue:
        return_queue.put(tracer.trace)
    else:
        return tracer.trace


def approx_svp_time(seed, params, return_queue=None, progressive=False):
    """Run Approx-SVP_{1.05} reduction on ``A`` using ``params``.

    :param seed: random seed for matrix creation
    :param params: BKZ preprocessing parameters, preprocessing block size is ignored
    :param return_queue: if not ``None``, the result is put on this queue.
    :param progressive: run Progressive-BKZ

    """
    from chal import load_svp_challenge
    from fpylll.algorithms.bkz import BKZReduction as BKZBase

    FPLLL.set_random_seed(seed)
    A = load_svp_challenge(params.block_size, seed=seed)
    M = GSO.Mat(A)
    M.update_gso()

    gh = gaussian_heuristic(M.r())
    target_norm = 1.05**2 * gh

    nodes_per_second = 2.0 * 10**9 / 100.0

    self = BKZ2(M)
    tracer = BKZTreeTracer(self, start_clocks=True)

    rerandomize = False
    preproc_cost = None
    with tracer.context(("tour", 0)):
        while M.get_r(0, 0) > target_norm:
            with tracer.context("preprocessing"):
                if rerandomize:
                    self.randomize_block(1, params.block_size,
                                         density=params.rerandomization_density, tracer=tracer)
                with tracer.context("reduction"):
                    BKZBase.svp_preprocessing(self, 0, params.block_size, params, tracer)  # LLL
                    preproc = round(0.9878*params.block_size - 24.12)  # curve fitted to chal.py output
                    prepar = params.__class__(block_size=preproc, strategies=params.strategies, flags=BKZ.GH_BND)
                    self.tour(prepar, 0, params.block_size, tracer=tracer)

            if preproc_cost is None:
                preproc_cost = float(tracer.trace.find("preprocessing")["walltime"])
                preproc_cost *= nodes_per_second

            with tracer.context("pruner"):
                step_target = M.get_r(0, 0)*0.99 if progressive else target_norm
                pruner = Pruning.Pruner(step_target, preproc_cost, [M.r()],
                                        target=1, metric=Pruning.EXPECTED_SOLUTIONS)
                coefficients = pruner.optimize_coefficients([1.]*M.d)
            try:
                enum_obj = Enumeration(self.M)
                with tracer.context("enumeration", enum_obj=enum_obj, full=True):
                    max_dist, solution = enum_obj.enumerate(0, params.block_size, target_norm, 0,
                                                            pruning=coefficients)[0]
                with tracer.context("postprocessing"):
                    self.svp_postprocessing(0, params.block_size, solution, tracer=tracer)
                rerandomize = False
            except EnumerationError:
                rerandomize = True

            self.M.update_gso()
            logger.debug("r_0: %7.2f, target: %7.2f, preproc: %3d"%(log(M.get_r(0, 0), 2),
                                                                    log(target_norm, 2), preproc))

    tracer.exit()
    tracer.trace.data["|A_0|"] = A[0].norm()
    tracer.trace.data["preprocessing_block_size"] = preproc

    if return_queue:
        return_queue.put(tracer.trace)
    else:
        return tracer.trace


@begin.start(auto_convert=True)
@begin.logging
@begin.convert(max_block_size=int, lower_bound=int, step_size=int, c=float)
def call(max_block_size: "compute up to this block size",
         strategies: "BKZ strategies",
         dump_filename: """results are stored in this filename, if ``None``
         then ``data/fplll-estimates-{lattice_type}.sobj`` is used.""" = None,
         npexp: "number of experiments to run parallel" = 4,
         ncores: "number of cores to use per experiment" = 1,
         algorithm: "one of SVP, oSVP or HSVP1.05" = "SVP",
         progressive: "use Progressive-BKZ in Approx-SVP" = False,
         lower_bound: "Start experiment in this dimension" = None,
         step_size: "Increment dimension by this much each iteration" = 2,
         c: "Overshooting parameter (for oSVP)" = 0.25,
         samples=48):
    """
Run (Approx-)SVP reduction and record statistics.

    """
    results = OrderedDict()

    FPLLL.set_threads(ncores)

    if dump_filename is None:
        dump_filename = "../data/fplll-observations,{lattice_type},[{strategies}].sobj".format(
            strategies=os.path.basename(strategies),
            lattice_type="svp-challenge" if algorithm == "HSVP1.05" is False else "qary")

    if isinstance(strategies, str):
        if strategies.endswith(".json"):
            strategies = load_strategies_json(bytes(strategies, "ascii"))
        elif strategies.endswith(".sobj"):
            strategies = pickle.load(open(strategies, "rb"))

    if algorithm.lower() == "svp":
        target = svp_time
        lower_bound = lower_bound if lower_bound else 20
    elif algorithm.lower() == "hsvp1.05":
        target = approx_svp_time
        lower_bound = lower_bound if lower_bound else 60
    elif algorithm.lower() == "osvp":
        target = osvp_time
        lower_bound = lower_bound if lower_bound else 20
    else:
        raise ValueError("Algorithm '%s' not known."%algorithm)

    for block_size in range(lower_bound, max_block_size+1, step_size):
        return_queue = Queue()
        result = OrderedDict([("total time", None)])

        traces = []
        # 2. run `k` processes in parallel
        for chunk in chunk_iterator(range(samples), npexp):
            processes = []
            for i in chunk:
                seed = i
                param = BKZ.Param(block_size=block_size, strategies=list(strategies), flags=BKZ.VERBOSE|BKZ.GH_BND)
                param["c"] = c
                if npexp > 1:
                    process = Process(target=target, args=(seed, param, return_queue))
                    processes.append(process)
                    process.start()
                else:
                    traces.append(target(seed, param, None))

            active_children()

            if npexp > 1:
                for process in processes:
                    traces.append(return_queue.get())

        preprocessing_block_size =  sum([trace.data["preprocessing_block_size"] for trace in traces])/samples
        total_time = sum([float(trace.data["walltime"]) for trace in traces])/samples
        length     = sum([trace.data["|A_0|"] for trace in traces])/samples
        enum_nodes = sum([sum([float(enum["#enum"]) for enum in trace.find_all("enumeration")])
                          for trace in traces])/samples

        logger.info("= block size: %3d, m: %3d, t: %10.3fs, log(#enum): %6.1f |A_0| = 2^%.1f",
                    block_size, samples, total_time, log(enum_nodes, 2), log(length, 2))

        result["total time"] = total_time
        result["betaprime"] = preprocessing_block_size
        result["length"] = length
        result["#enum"] = enum_nodes
        result["traces"] = traces

        results[block_size] = result

        if results[block_size]["total time"] > 1.0 and samples > max(8, npexp):
            samples //= 2
            if samples < npexp:
                samples = npexp

        pickle.dump(results, open(dump_filename, "wb"))

    return results
