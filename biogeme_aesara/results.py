# results.py

import aesara
import dill as pickle
import numpy as np
import pandas as pd

from numpy import nan_to_num as nan2num

from biogeme_aesara.functions import hessians, bhhh
from biogeme_aesara.statistics import *


class Results:
    def __init__(self, model, db):
        """ Generate the output results to stdout

        Args:
            model (Class): The model class object or pickle file
            db (Class): The database class object
        """
        if isinstance(model, str):
            with open("best_model.pkl", "rb") as f:
                model = pickle.load(f)

        sample_size = len(db.data)
        n_params = len([p for p in model.params if p.status != 1])

        null_loglike = model.null_ll
        max_loglike = model.best_ll

        self.beta_results = get_beta_statistics(model, db)
        self.rho_square = nan2num(1.0 - max_loglike / null_loglike)
        self.rho_square_bar = nan2num(1.0 - (max_loglike - n_params) / null_loglike)

        self.akaike = 2.0 * (n_params - max_loglike)
        self.bayesian = -2.0 * max_loglike + n_params * np.log(sample_size)
        self.g_norm = gradient_norm(model, db)
        self.correlationMatrix = correlation_matrix(model, db)
        self.model = model
        self.n_params = n_params
        self.sample_size = sample_size
        self.best_epoch = model.best_epoch
        self.best_error = model.best_error
        self.null_loglike = null_loglike
        self.max_loglike = max_loglike

        self.print_results = (
            "Number of parameters: {}\n".format(self.n_params)
            + "Sample size: {}\n".format(self.sample_size)
            + "Null loglikelihood: {0:.6f}\n".format(self.null_loglike)
            + "Final loglikelihood: {0:.6f}\n".format(self.max_loglike)
            + "Validation error: {0:.3f}%\n".format(100 * self.best_error)
            + "Rho square (null): {0:.3f}\n".format(self.rho_square)
            + "Rho bar square (null): {0:.3f}\n".format(self.rho_square_bar)
            + "AIC: {0:.2f}\n".format(self.akaike)
            + "BIC: {0:.2f}\n".format(self.bayesian)
            + "Final gradient norm: {0:.3f}\n".format(self.g_norm)
        )

        print(self.print_results)
        print(self.beta_results, "\n")


def get_beta_statistics(model, db):
    H = aesara.function(
        inputs=model.inputs,
        outputs=hessians(model.p_y_given_x, model.y, model.params),
        on_unused_input="ignore",
    )

    BHHH = aesara.function(
        inputs=model.inputs,
        outputs=bhhh(model.p_y_given_x, model.y, model.params),
        on_unused_input="ignore",
    )

    h = H(*db.input_data())
    bh = BHHH(*db.input_data())

    pandas_stats = pd.DataFrame(
        columns=[
            "Value",
            "Std err",
            "t-test",
            "p-value",
            "Rob. Std err",
            "Rob. t-test",
            "Rob. p-value",
        ],
        index=[p.name for p in model.params if p.status != 1],
    )
    stderr = stderror(h, model.params)
    robstderr = rob_stderror(h, bh, model.params)
    pandas_stats["Std err"] = stderr
    pandas_stats["t-test"] = t_test(stderr, model.params)
    pandas_stats["p-value"] = p_value(stderr, model.params)
    pandas_stats["Rob. Std err"] = robstderr
    pandas_stats["Rob. t-test"] = t_test(robstderr, model.params)
    pandas_stats["Rob. p-value"] = p_value(robstderr, model.params)
    pandas_stats = pd.DataFrame(index=[p.name for p in model.params]).join(pandas_stats)

    pandas_stats["Value"] = [p() for p in model.params]

    return pandas_stats.sort_index().round(6).fillna("-").astype("O")
