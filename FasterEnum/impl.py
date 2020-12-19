# -*- coding: utf-8 -*-
from fpylll.algorithms.bkz2 import BKZReduction as BKZBase
from fpylll.tools.bkz_stats import dummy_tracer
from fpylll import BKZ, Enumeration, EnumerationError
from math import ceil, log


class BKZReduction(BKZBase):
    def __init__(self, A, c=0.25):
        """Create new BKZ object.

        :param A: an integer matrix, a GSO object or an LLL object
        :param c: overshoot parameter

        """
        BKZBase.__init__(self, A)
        self.c = c

    def svp_preprocessing(
        self, kappa, end, block_size, params, tracer=dummy_tracer
    ):
        clean = True

        lll_start = kappa if params.flags & BKZ.BOUNDED_LLL else 0
        with tracer.context("lll"):
            self.lll_obj(lll_start, lll_start, end)
            if self.lll_obj.nswaps > 0:
                clean = False

        if not params.strategies[block_size].preprocessing_block_sizes:
            return clean

        for preproc in params.strategies[block_size].preprocessing_block_sizes:
            prepar = params.__class__(
                block_size=preproc,
                strategies=params.strategies,
                flags=BKZ.GH_BND,
            )
            clean &= self.tour(prepar, kappa, end, tracer=tracer)
        return clean

    def svp_reduction(self, kappa, block_size, params, tracer=dummy_tracer):
        """

        :param kappa:
        :param block_size:
        :param params:
        :param tracer:

        """

        self.lll_obj.size_reduction(0, kappa + 1)
        old_first, old_first_expo = self.M.get_r_exp(kappa, kappa)

        remaining_probability, rerandomize = 1.0, False

        # NOTE: In the tail we might have less than (1+c)β space
        end = min(ceil(kappa + (1 + self.c) * block_size), self.M.d)

        while remaining_probability > 1.0 - params.min_success_probability:
            with tracer.context("preprocessing"):
                if rerandomize:
                    with tracer.context("randomization"):
                        self.randomize_block(
                            kappa + 1,
                            kappa + block_size,
                            density=params.rerandomization_density,
                            tracer=tracer,
                        )
                with tracer.context("reduction"):
                    self.svp_preprocessing(
                        kappa, end, block_size, params, tracer=tracer
                    )
            with tracer.context("pruner"):
                radius, exp, pruning = self.get_pruning(
                    kappa, block_size, params, tracer
                )

            try:
                enum_obj = Enumeration(self.M)
                with tracer.context(
                    "enumeration",
                    enum_obj=enum_obj,
                    probability=pruning.expectation,
                    full=True,
                ):  # HACK: we wan to record all enum costs.
                    max_dist, solution = enum_obj.enumerate(
                        kappa,
                        kappa + block_size,
                        radius,
                        exp,
                        pruning=pruning.coefficients,
                    )[0]
                with tracer.context("postprocessing"):
                    self.svp_postprocessing(
                        kappa, block_size, solution, tracer=tracer
                    )
                rerandomize = False

            except EnumerationError:
                rerandomize = True

            remaining_probability *= 1 - pruning.expectation

        self.lll_obj.size_reduction(0, kappa + 1)
        new_first, new_first_expo = self.M.get_r_exp(kappa, kappa)

        clean = old_first <= new_first * 2 ** (new_first_expo - old_first_expo)
        return clean

    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
        if max_row == -1:
            max_row = self.A.nrows

        clean = True

        limit = int(ceil((1 + self.c) * params.block_size))

        for kappa in range(min_row, max_row - limit):
            # We run SVP reductions with cost β^{β/8 + o(β)}
            assert max_row - kappa >= params.block_size
            clean &= self.svp_reduction(
                kappa, params.block_size, params, tracer=tracer
            )

        cost_ceil = log(params.block_size) * params.block_size / 8.0

        for i, kappa in enumerate(range(max_row - limit, max_row - 1)):
            # we reduce the block size roughly by one every second index to maintain the average case complexity
            block_size = params.block_size - int(ceil((i + 1) / 2))
            # if worst case at the local block size < average case at global block size, then might as well use that.
            while cost_ceil > 0.184 * log(block_size + 1) * (block_size + 1):
                block_size += 1
            block_size = min(max_row - kappa, block_size)
            clean &= self.svp_reduction(
                kappa, block_size, params, tracer=tracer
            )
        return clean
