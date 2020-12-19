#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate BKZ variants.
"""

import logging
import pickle
from collections import OrderedDict
from math import ceil, lgamma, log, pi
from contextlib import contextmanager

import begin
from fpylll import BKZ, GSO, IntegerMatrix, LLL, Pruning, load_strategies_json
from fpylll.tools.bkz_stats import Accumulator, Node, Tracer, dummy_tracer, pretty_dict
from fpylll.tools.quality import basis_quality
from fpylll.util import gaussian_heuristic

# Verbose logging

logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S %Z")

# define a Handler which writes INFO messages or higher to the sys.stderr
logger = logging.getLogger(__name__)


def gh_normalizer_log2(d, plain=False):
    """
    Return the log2() of normalization factor for the Gaussian heuristic.

    :param d: dimensions
    :param plain: if ``True`` do not deviate from plain formula for small dimensions

    """

    rk = (
        0.789527997160000,
        0.780003183804613,
        0.750872218594458,
        0.706520454592593,
        0.696345241018901,  # noqa
        0.660533841808400,
        0.626274718790505,
        0.581480717333169,
        0.553171463433503,
        0.520811087419712,
        0.487994338534253,
        0.459541470573431,
        0.414638319529319,
        0.392811729940846,
        0.339090376264829,
        0.306561491936042,
        0.276041187709516,
        0.236698863270441,
        0.196186341673080,
        0.161214212092249,
        0.110895134828114,
        0.0678261623920553,
        0.0272807162335610,
        -0.0234609979600137,
        -0.0320527224746912,
        -0.0940331032784437,
        -0.129109087817554,
        -0.176965384290173,
        -0.209405754915959,
        -0.265867993276493,
        -0.299031324494802,
        -0.349338597048432,
        -0.380428160303508,
        -0.427399405474537,
        -0.474944677694975,
        -0.530140672818150,
        -0.561625221138784,
        -0.612008793872032,
        -0.669011014635905,
        -0.713766731570930,
        -0.754041787011810,
        -0.808609696192079,
        -0.859933249032210,
        -0.884479963601658,
        -0.886666930030433,
    )

    if plain or d > 45:
        log_vol = log(pi, 2) / 2 * d - lgamma(d / 2.0 + 1) / log(2.0)
    else:
        log_vol = -rk[-d] * d + sum(rk[-d:])
    return log_vol


class BKZSimulationTreeTracer(Tracer):
    """
    A tracer for tracing simulations.
    """

    def __init__(self, instance, verbosity=False, root_label="bkz"):
        """
        Create a new tracer instance.

        :param instance: BKZ-like object instance
        :param verbosity: print information, integers >= 0 are also accepted
        :param root_label: label to give to root node

        """

        Tracer.__init__(self, instance, verbosity)
        self.trace = Node(root_label)
        self.current = self.trace

    def enter(self, label, **kwds):
        """
        Enter new context with label

        :param label: label

        """
        self.current = self.current.child(label)
        self.reenter()

    def reenter(self, **kwds):
        """
        Reenter current context.
        """
        self.current.data["cost"] = self.current.data.get("cost", Accumulator(1, repr="sum"))

    def inc_cost(self, cost):
        self.current.data["cost"] += Accumulator(cost, repr="sum")

    def exit(self, **kwds):
        """
        When the label is a tour then the status is printed if verbosity > 0.
        """
        node = self.current
        label = node.label

        if label[0] == "tour":
            data = basis_quality([2 ** (2 * r_) for r_ in self.instance.r])
            for k, v in data.items():
                if k == "/":
                    node.data[k] = Accumulator(v, repr="max")
                else:
                    node.data[k] = Accumulator(v, repr="min")

        if self.verbosity and label[0] == "tour":
            report = OrderedDict()
            report["i"] = label[1]
            report["#enum"] = node.sum("#enum")
            report["r_0"] = node["r_0"]
            report["/"] = node["/"]
            print(pretty_dict(report))

        self.current = self.current.parent


class BKZQualitySimulation(object):
    """
    Simulate quality of BKZ reduction.
    """

    def __init__(self, A, preprocessing_levels=1, preprocessing_cutoff=45):
        """
        Create a new BKZ Simulation object.

        :param A: An integer matrix, a GSO object or a list of squared Gram-Schmidt norms.
        :param preprocessing_levels: how many levels of preprocessing to simulate (slow!)

        .. note :: Our internal representation is log2 norms of Gram-Schmidt vectors (not squared).

        """
        if isinstance(A, GSO.Mat):
            A.update_gso()
            r = A.r()
            self.r = [log(r_, 2) / 2.0 for r_ in r]
        elif isinstance(A, LLL.Reduction):
            A.M.update_gso()
            r = A.M.r()
            self.r = [log(r_, 2) / 2.0 for r_ in r]
        elif isinstance(A, IntegerMatrix):
            M = GSO.Mat(LLL.reduction(A))
            M.update_gso()
            r = M.r()
            self.r = [log(r_, 2) / 2.0 for r_ in r]
        else:
            try:
                self.r = [log(r_, 2) / 2.0 for r_ in A]
            except TypeError:
                raise TypeError("Unsupported type '%s'" % type(A))
        self.preprocessing_levels = preprocessing_levels
        self.preprocessing_cutoff = preprocessing_cutoff

    @contextmanager
    def descent(self, preproc):
        """
        Context for limiting decent downward when preprocessing.
        """
        if not hasattr(self, "level"):
            self.level = 0

        self.level += 1
        skip = (preproc <= self.preprocessing_cutoff) or (self.preprocessing_levels <= self.level)
        try:
            yield skip
        finally:
            self.level -= 1

    def __call__(self, params, min_row=0, max_row=-1, tracer=None, **kwds):
        """
        Simulate quality of BKZ reduction.

        :param params: BKZ parameters
        :param min_row: start processing at min_row (inclusive)
        :param max_row: stop processing at max_row (exclusive)
        :param kwds: added to parameters
        :returns: Squared Gram-Schmidt norms.

        """
        self.level = 0
        i = 0

        if tracer is None:
            tracer = dummy_tracer

        params = params.new(**kwds)

        while True:
            with tracer.context(("tour", i)):
                clean = self.tour(params, min_row, max_row, tracer=tracer)
            i += 1
            if clean or params.block_size >= len(self.r):
                break
            if (params.flags & BKZ.MAX_LOOPS) and i >= params.max_loops:
                break

        return tuple([2 ** (2 * r_) for r_ in self.r])

    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
        """
        One tour of BKZ.

        :param params: BKZ parameters
        :param min_row: start processing at min_row (inclusive)
        :param max_row: stop processing at max_row (exclusive)
        :returns: whether the basis remained untouched or not

        """
        if max_row == -1:
            max_row = len(self.r)

        clean = True

        for kappa in range(min_row, max_row - 1):
            block_size = min(params.block_size, max_row - kappa)
            clean &= self.svp_reduction(kappa, block_size, params, tracer=tracer)

        return clean

    def svp_preprocessing(self, kappa, end, block_size, params, tracer=dummy_tracer):
        """

        :param kappa:
        :param end:
        :param block_size:
        :param params:
        :param tracer:

        """
        clean = True

        if not params.strategies[block_size].preprocessing_block_sizes:
            return clean

        for preproc in params.strategies[block_size].preprocessing_block_sizes:
            with self.descent(preproc) as skip:
                if not skip:
                    prepar = params.new(block_size=preproc, flags=BKZ.GH_BND)
                    clean &= self.tour(prepar, kappa, end, tracer=tracer)

        return clean

    def svp_call(self, kappa, block_size, params, hkz=True, tracer=dummy_tracer):
        """
        Return log norm as predicted by Gaussian heuristic.

        :param kappa: SVP start index
        :param block_size: SVP dimension
        :param params: ignored
        :param hkz: assume the call happens inside the HKZ part of a basis
        :param tracer: ignored

        """
        log_vol = sum(self.r[kappa : kappa + block_size])
        normalizer = gh_normalizer_log2(block_size, plain=not hkz)
        return (log_vol - normalizer) / block_size

    def svp_postprocessing(self, kappa, block_size, solution, tracer=dummy_tracer):
        """
        Insert vector and distribute additional weight equally.

        :param kappa: SVP start index
        :param block_size:  SVP dimension
        :param solution: norm of the found vector
        :param tracer: ignored

        """
        clean = True
        if solution < self.r[kappa]:
            clean = False
            delta = (self.r[kappa] - solution) / (block_size - 1)
            self.r[kappa] = solution
            for j in range(kappa + 1, kappa + block_size):
                self.r[j] += delta

        return clean

    def svp_reduction(self, kappa, block_size, params, tracer=dummy_tracer):
        """
        Preprocessing, oracle call, postprocessing

        :param kappa: SVP start index
        :param block_size: SVP dimension
        :param params: BKZ parameters
        :param tracer: tracer object

        """
        clean = True

        with tracer.context("preprocessing"):
            clean &= self.svp_preprocessing(kappa, kappa + block_size, block_size, params, tracer=tracer)

        with tracer.context("enumeration"):
            solution = self.svp_call(kappa, block_size, params, hkz=block_size != params.block_size, tracer=tracer)

        with tracer.context("postprocessing"):
            clean &= self.svp_postprocessing(kappa, block_size, solution, tracer=tracer)

        return clean


class ProcrastinatingBKZQualitySimulation(BKZQualitySimulation):
    """
    Simulate quality of Procrastinating-BKZ reduction.
    """

    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
        """

        :param params: BKZ parameters
        :param min_row: start processing at min_row (inclusive)
        :param max_row: stop processing at max_row (exclusive)
        :returns: whether the basis remained untouched or not

        """
        if max_row == -1:
            max_row = len(self.r)

        clean = True
        limit = int(ceil((1 + params["c"]) * params.block_size))

        # We run SVP reductions with cost β^{β/8 + o(β)}
        for kappa in range(min_row, max_row - limit):
            clean &= self.svp_reduction(kappa, params.block_size, params, tracer=tracer)

        # NOTE: Could pick 0.11 here, too
        cost_ceil = log(params.block_size) * params.block_size / 8.0

        for i, kappa in enumerate(range(max_row - limit, max_row - 1)):
            # we reduce the block size roughly by one every second index to maintain the average
            # case complexity
            block_size = params.block_size - int(ceil((i + 1) / 2))
            # if worst case at the local block size < average case at global block size, then might
            # as well use that.
            while cost_ceil > 0.184 * log(block_size + 1) * (block_size + 1):
                block_size += 1
            block_size = min(max_row - kappa, block_size)
            clean &= self.svp_reduction(kappa, block_size, params, tracer=tracer)

        return clean

    def svp_reduction(self, kappa, block_size, params, tracer=dummy_tracer):
        """
        Preprocessing, oracle call, postprocessing

        :param kappa: SVP start index
        :param block_size: SVP dimension
        :param params: BKZ parameters
        :param tracer: tracer object

        """
        # We extend the preprocessing area beyond kappa+beta
        end = ceil(kappa + (1 + params["c"]) * block_size)
        if end > len(self.r):
            if not block_size < params.block_size:
                raise ValueError("Bug: trying to access index %d" % end)
            else:
                end = len(self.r)

        clean = True

        with tracer.context("preprocessing"):
            clean &= self.svp_preprocessing(kappa, end, block_size, params, tracer=tracer)

        with tracer.context("enumeration"):
            solution = self.svp_call(kappa, block_size, params, hkz=kappa + block_size == end, tracer=tracer)
        with tracer.context("postprocessing"):
            clean &= self.svp_postprocessing(kappa, block_size, solution, tracer=tracer)

        return clean


class SDProcrastinatingBKZQualitySimulation(ProcrastinatingBKZQualitySimulation):
    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
        d = len(self.r)
        if max_row == -1:
            max_row = d

        self.r = [-r_ for r_ in self.r[::-1]]
        ProcrastinatingBKZQualitySimulation.tour(self, params, min_row=d - max_row, max_row=d - min_row)
        self.r = [-r_ for r_ in self.r[::-1]]
        ProcrastinatingBKZQualitySimulation.tour(self, params, min_row=min_row, max_row=max_row)


class BKZSimulation(BKZQualitySimulation):
    """
    Simulate quality and cost of Procrastinating-BKZ reduction.
    """

    def __init__(self, A):
        """
        Create a new simulation object

        :param A: An integer matrix, a GSO object or a list of squared Gram-Schmidt norms.

        """
        super(BKZSimulation, self).__init__(A, preprocessing_levels=1024, preprocessing_cutoff=10)

    def __call__(self, params, min_row=0, max_row=-1, **kwds):
        """

        :param params: BKZ parameters
        :param min_row: start processing at min_row (inclusive)
        :param max_row: stop processing at max_row (exclusive)
        :returns: Squared Gram-Schmidt norms and cost in enumeration nodes

        """
        tracer = BKZSimulationTreeTracer(self, verbosity=params.flags & BKZ.VERBOSE)
        r = super().__call__(params, min_row, max_row, tracer, **kwds)
        tracer.exit()
        self.trace = tracer.trace
        return r

    def get_pruning(self, kappa, block_size, params, tracer=dummy_tracer):
        strategy = params.strategies[block_size]
        radius = 2 ** self.r[kappa]
        gh_radius = gaussian_heuristic([2 ** r_ for r_ in self.r[kappa : kappa + block_size]])

        if params.flags & BKZ.GH_BND and block_size > 30:
            radius = min(radius, gh_radius)  # HACK

        return radius, strategy.get_pruning(radius, gh_radius)

    def lll(self, start, end, tracer=dummy_tracer):
        """
        Simulate LLL on ``r[start:end]``

        :param start: first index to be touched
        :param end: last index to be touched (exclusive)

        """
        d = end - start
        if d <= 1:
            return

        cost = d ** 3
        delta_0 = log(1.0219, 2)
        alpha = delta_0 * (-2 * d / float(d - 1))
        rv = sum(self.r[start:end]) / d

        self.r[start:end] = [2 * (i * alpha + delta_0 * d) + rv for i in range(d)]

        try:
            tracer.inc_cost(cost)
        except AttributeError:
            pass

    def svp_preprocessing(self, kappa, end, block_size, params, tracer=dummy_tracer):
        """
        """
        with tracer.context("lll"):
            self.lll(kappa + 1, kappa + block_size, tracer)

        return super().svp_preprocessing(kappa, end, block_size, params, tracer)

    def svp_reduction(self, kappa, block_size, params, tracer=dummy_tracer):
        """
        Preprocessing, oracle call, postprocessing

        :param kappa: SVP start index
        :param block_size: SVP dimension
        :param params: BKZ parameters
        :param tracer: tracer object

        """

        remaining_probability = 1.0
        clean = True

        while remaining_probability > 1.0 - params.min_success_probability:
            with tracer.context("preprocessing"):
                clean &= self.svp_preprocessing(kappa, kappa + block_size, block_size, params, tracer=tracer)
                preproc_cost = tracer.current.sum("cost")

            with tracer.context("enumeration"):
                radius, pr = self.get_pruning(kappa, block_size, params, tracer)
                pruner = Pruning.Pruner(
                    radius, preproc_cost, [[2 ** r_ for r_ in self.r[kappa : kappa + block_size]]], target=0.51
                )

                solution = self.svp_call(kappa, block_size, params, hkz=params.block_size != block_size, tracer=tracer)
                try:
                    tracer.inc_cost(pruner.single_enum_cost(pr.coefficients))
                except AttributeError:
                    pass

            remaining_probability *= 1 - pr.expectation

            with tracer.context("postprocessing"):
                clean &= self.svp_postprocessing(kappa, block_size, solution, tracer=tracer)

        return clean


class ProcrastinatingBKZSimulation(BKZSimulation):
    """
    A simulator simulating both quality and time.
    """

    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
        """

        :param params: BKZ parameters
        :param min_row: start processing at min_row (inclusive)
        :param max_row: stop processing at max_row (exclusive)
        :returns: whether the basis remained untouched or not

        """
        if max_row == -1:
            max_row = len(self.r)

        clean = True
        limit = int(ceil((1 + params["c"]) * params.block_size))

        for kappa in range(min_row, max_row - limit):
            # We run SVP reductions with cost β^{β/8 + o(β) + o(d)}
            assert max_row - kappa >= params.block_size
            clean &= self.svp_reduction(kappa, params.block_size, params, tracer=tracer)

        cost_ceil = log(params.block_size) * params.block_size / 8.0

        for i, kappa in enumerate(range(max_row - limit, max_row - 1)):
            # we reduce the block size roughly by one every second index to maintain the average
            # case complexity
            block_size = params.block_size - int(ceil((i + 1) / 2))
            # if worst case at the local block size < average case at global block size, then might
            # as well use that.
            while cost_ceil > 0.184 * log(block_size + 1) * (block_size + 1):
                block_size += 1
            block_size = min(max_row - kappa, block_size)
            clean &= self.svp_reduction(kappa, block_size, params, tracer=tracer)
        return clean

    def svp_reduction(self, kappa, block_size, params, tracer=dummy_tracer):
        """
        Preprocessing, oracle call, postprocessing

        :param kappa: SVP start index
        :param block_size: SVP dimension
        :param params: BKZ parameters
        :param tracer: tracer object

        """
        remaining_probability = 1.0
        clean = True

        end = ceil(kappa + (1 + params["c"]) * block_size)
        if end > len(self.r):
            if not block_size < params.block_size:
                raise ValueError("Bug: trying to access index %d" % end)
            else:
                end = len(self.r)

        while remaining_probability > 1.0 - params.min_success_probability:
            with tracer.context("preprocessing"):
                clean &= self.svp_preprocessing(kappa, end, block_size, params, tracer=tracer)
                preproc_cost = tracer.current.sum("cost")

            with tracer.context("enumeration"):
                radius, pr = self.get_pruning(kappa, block_size, params, tracer)
                pruner = Pruning.Pruner(
                    radius, preproc_cost, [[2 ** r_ for r_ in self.r[kappa : kappa + block_size]]], target=0.51
                )

                solution = self.svp_call(kappa, block_size, params, hkz=kappa + block_size == end, tracer=tracer)

                tracer.inc_cost(pruner.single_enum_cost(pr.coefficients))

            remaining_probability *= 1 - pr.expectation

            with tracer.context("postprocessing"):
                clean &= self.svp_postprocessing(kappa, block_size, solution, tracer=tracer)

        return clean


def bkz_simulatef(cls, init_kwds=None, call_kwds=None):
    """
    Turn simulation class into a callable.

    :param cls: a Simulation class
    :param init_kwds: keywords passed to ``__init__``
    :param call_kwds: keywords passed to ``__call__``

    """

    if init_kwds is None:
        init_kwds = {}
    if call_kwds is None:
        call_kwds = {}

    def bkz_simulate(r, params):
        bkz = cls(r, **init_kwds)
        r = bkz(params, **call_kwds)
        return r, None

    return bkz_simulate


def svp_time(seed, params, return_queue=None):
    """Run SVP reduction on ``A`` using ``params``.

    :param seed: random seed for matrix creation
    :param params: BKZ parameters
    :param return_queue: if not ``None``, the result is put on this queue.

    """
    from cost import sample_r

    r = sample_r(params.block_size)
    bkz = BKZSimulation(r)
    tracer = BKZSimulationTreeTracer(bkz)

    with tracer.context(("tour", 0)):
        bkz.svp_reduction(0, params.block_size, params, tracer)

    tracer.exit()
    r = tuple([2 ** (r_) for r_ in bkz.r])

    tracer.trace.data["|A_0|"] = r[0]
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

    A = sample_matrix(ceil(params.block_size * (1 + params["c"])), seed=seed)
    M = GSO.Mat(A)
    bkz = ProcrastinatingBKZSimulation(M)
    tracer = BKZSimulationTreeTracer(bkz)

    with tracer.context(("tour", 0)):
        bkz.svp_reduction(0, params.block_size, params, tracer)

    tracer.exit()
    r = tuple([2 ** (r_) for r_ in bkz.r])

    tracer.trace.data["|A_0|"] = r[0]
    ppbs = params.strategies[params.block_size].preprocessing_block_sizes
    tracer.trace.data["preprocessing_block_size"] = ppbs[0] if ppbs else 2

    if return_queue:
        return_queue.put(tracer.trace)
    else:
        return tracer.trace


@begin.start(auto_convert=True)
@begin.logging
@begin.convert(max_block_size=int, lower_bound=int, step_size=int, c=float)
def call(
    max_block_size: "compute up to this block size",
    strategies: "BKZ strategies",
    algorithm: "one of SVP or oSVP" = "SVP",
    lower_bound: "Start experiment in this dimension" = None,
    step_size: "Increment dimension by this much each iteration" = 2,
    c: "Overshooting parameter (for oSVP)" = 0.25,
):
    """
    Simulate SVP reduction and record statistics.

    """
    if isinstance(strategies, str):
        if strategies.endswith(".json"):
            strategies = load_strategies_json(bytes(strategies, "ascii"))
        elif strategies.endswith(".sobj"):
            strategies = pickle.load(open(strategies, "rb"))

    if algorithm.lower() == "svp":
        target = svp_time
        lower_bound = lower_bound if lower_bound else 20
    elif algorithm.lower() == "osvp":
        target = osvp_time
        lower_bound = lower_bound if lower_bound else 20
    else:
        raise ValueError("Algorithm '%s' not known." % algorithm)

    for block_size in range(lower_bound, max_block_size + 1, step_size):
        param = BKZ.Param(block_size=block_size, strategies=list(strategies), c=c, flags=BKZ.VERBOSE | BKZ.GH_BND)
        trace = target(0, param, None)

        length = trace.data["|A_0|"]
        enum_nodes = float(trace.sum("cost"))

        logger.info(
            "= block size: %3d, log(#enum): %6.1f |A_0| = 2^%.1f", block_size, log(enum_nodes, 2), log(length, 2)
        )
