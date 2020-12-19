#!/usr/bin/env python
# -*- coding: utf-8 -*-
import begin
import csv
import pickle
import json
import logging
import math


@begin.subcommand
def svp_challenges(
    url: "SVP Challenges URL" = "https://www.latticechallenge.org/svp-challenge/halloffame.php",
    filename: "output filename." = None,
):
    """
    Download SVP Challenge data into json or sobj
    """
    from bs4 import BeautifulSoup
    import urllib.request
    from collections import namedtuple

    Record = namedtuple("Record", ["block_size", "norm", "seed", "contestant", "algorithm", "description", "cycles"])

    records = []

    with urllib.request.urlopen(url) as response:
        html = response.read()
        entries = BeautifulSoup(html, "html.parser").table.find_all("tr")
        for entry in entries[1:]:
            block_size = int(entry.find_all("td")[1].get_text())
            norm = int(entry.find_all("td")[2].get_text())
            seed = int(entry.find_all("td")[3].get_text())
            contestant = entry.find_all("td")[4].get_text()
            algorithm = entry.find_all("td")[6].get_text()
            description = entry.get("title").replace("<br />", "\n")
            records.append(
                Record(
                    block_size=block_size,
                    norm=norm,
                    seed=seed,
                    contestant=contestant,
                    algorithm=algorithm,
                    description=description,
                    cycles=None,
                )
            )
    records = tuple(records)
    if filename:
        if filename.endswith(".sobj"):
            pickle.dump(open(filename, "wb"), records)
        elif filename.endswith(".json"):
            json.dump([record._asdict() for record in records], open(filename, "w"), indent=4)
        else:
            raise ValueError("Filetype of '%s' not supported." % filename)
    return records


def read_csv(filename, columns, read_range=None, ytransform=lambda y: y):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        data = []
        for i, row in enumerate(reader):
            if i == 0:
                columns = row.index(columns[0]), row.index(columns[1])
                continue
            data.append((int(row[columns[0]]), ytransform(float(row[columns[1]]))))

    if read_range is not None:
        data = [(x, y) for x, y in data if x in read_range]
    data = sorted(data)
    logging.debug(data)
    X = [x for x, y in data]
    Y = [y for x, y in data]
    return tuple(X), tuple(Y)


@begin.subcommand
@begin.convert(low_index=int, high_index=int, columns=tuple, leading_coefficient=float)
def cost_fit(
    filename: ".csv filename to fit",
    low_index: "start fitting at this index" = 50,
    high_index: "stop fitting at this index (exclusive)" = 100,
    columns: "csv columns to select" = ("d", "total cost"),
    leading_coefficient: "leading coefficient of expression" = None,
):
    import numpy as np
    from math import log
    from scipy.optimize import curve_fit

    X, Y = read_csv(
        filename, columns=columns, read_range=range(low_index, high_index), ytransform=lambda y: log(y, 2.0)
    )

    if leading_coefficient is None:

        def f(x, a, b, c):
            return a * x * np.log2(x) + b * x + c

    else:

        def f(x, b, c):
            return leading_coefficient * x * np.log2(x) + b * x + c

    r = list(curve_fit(f, X, Y)[0])
    if leading_coefficient is not None:
        r = [leading_coefficient] + r
    logging.info("{r[0]:.4}*x*log(x,2) + {r[1]:.3}*x + {r[2]:.4}".format(r=r))
    return r


@begin.subcommand
@begin.convert(low_index=int, high_index=int, columns=tuple)
def preproc_fit(
    filename: ".csv filename to fit",
    low_index: "start fitting at this index" = 50,
    high_index: "stop fitting at this index (exclusive)" = 100,
    columns: "csv columns to select" = ("d", "betaprime"),
):
    """
    """
    from scipy.optimize import curve_fit

    def f(x, a, b):
        return a * x + b

    X, Y = read_csv(filename, columns=columns, read_range=range(low_index, high_index))
    r = tuple(curve_fit(f, X, Y)[0])
    print(r)
    return r

@begin.subcommand
@begin.convert(low_index=int, high_index=int, columns=tuple)
def prob_fit(
    filename: ".csv filename to fit",
    low_index: "start fitting at this index" = 50,
    high_index: "stop fitting at this index (exclusive)" = 100,
    columns: "csv columns to select" = ("d", "probability"),
):
    """
    """
    from scipy.optimize import curve_fit

    def f(x, a, b):
        return a * x + b

    X, Y = read_csv(filename, columns=columns, read_range=range(low_index, high_index),
                    ytransform=lambda y: math.log(y, 2))
    r = tuple(curve_fit(f, X, Y)[0])
    print(r)
    return r


# @begin.subcommand
# def gso_plots_sobj_csv(filename: ".sobj filename to convert"):
#     """
#     Convert output of ``./cost.py gso_plots`` to ``.csv`` suitable to produce LaTeX plots and tables.
#     """
#     plots = pickle.load(open(filename, "rb"))
#     mx = max(plots.keys())

#     with open(filename.replace(".sobj", ".csv"), "w", newline="") as csvfile:
#         csvwriter = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

#         csvwriter.writerow(["d"] + list(range(mx)))

#         for d in plots.keys():
#             csvwriter.writerow((d,) + plots[d] + tuple([None] * (mx - len(plots[d]))))


@begin.subcommand
def cost_sobj_csv(filename: ".sobj filename to convert"):
    """
    Convert output of ``./cost.py`` to ``.csv`` suitable to produce LaTeX plots and tables.
    """
    strategies, costs = pickle.load(open(filename, "rb"))

    with open(filename.replace(".sobj", ".csv"), "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow(["d", "betaprime", "total cost", "preprocessing", "single enum", "probability"])

        for beta in range(3, len(strategies)):
            strategy, cost = strategies[beta], costs[beta]
            betaprime = strategy.preprocessing_block_sizes
            betaprime = betaprime[0] if betaprime else 2
            csvwriter.writerow(
                [
                    beta,
                    betaprime,
                    round(cost["total cost"]),
                    round(cost["preprocessing"]),
                    round(cost["single enum"]),
                    cost["probability"],
                ]
            )


@begin.subcommand
def chal_sobj_csv(filename: ".sobj filename to convert"):
    """
    Convert output of ``./chal.py`` to ``.csv`` suitable to produce LaTeX plots and tables.
    """
    costs = pickle.load(open(filename, "rb"))

    with open(filename.replace(".sobj", ".csv"), "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow(["d", "betaprime", "total cost", "preprocessing", "single enum", "probability"])

        for beta, cost in costs.items():
            csvwriter.writerow(
                [
                    beta,
                    cost["preprocessing block size"],
                    round(cost["total cost"]),
                    round(cost["preprocessing"]),
                    round(cost["single enum"]),
                    cost["probability"],
                ]
            )


@begin.subcommand
def call_sobj_csv(filename: ".sobj filename to convert"):
    """
    Convert output of ``./call.py`` to ``.csv`` suitable to produce LaTeX plots and tables.
    """
    costs = pickle.load(open(filename, "rb"))

    with open(filename.replace(".sobj", ".csv"), "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow(["d", "betaprime", "total time", "total cost"])

        for beta, cost in costs.items():
            try:
                betaprime = cost["strategy"].preprocessing_block_sizes
                betaprime = betaprime[0] if betaprime else 2
            except KeyError:
                betaprime = int(cost["betaprime"])
            csvwriter.writerow([beta, betaprime, cost["total time"], round(cost["#enum"])])


# def compile_costs(strategies, lattice_type="qary"):
#     """
#     Compute costs for a given list of strategies
#     """

#     costs = [{"total cost": 0.0}] * 3
#     for block_size in range(3, len(strategies)):
#         costs.append(enumeration_cost(sample_r(block_size, lattice_type=lattice_type), strategies, costs))
#         logging.info("%3d :: %5.1f %s"%(block_size, log(costs[-1]["total cost"], 2), strategies[block_size]))
#     return costs


@begin.start
@begin.logging
def run():
    pass
