# __init__.py

from ast import Expr
import aesara
import aesara.tensor as aet
import dill as pickle
import numpy as np
import tqdm

from aesara.tensor.var import TensorVariable
from biogeme.database import Database
from biogeme.expressions import Beta

from .functions import full_loglikelihood, errors
from .utils import learn_rate_tempering

__version__ = "0.1.0"
floatX = aesara.config.floatX


class BaseModelClass:
    def __init__(self):
        pass

    def store_params(self, locals):
        for _, local in locals.items():
            if isinstance(local, BetaShared):
                # update model params into list
                self.params.append(local)

    def get_betas(self):
        return [p for p in self.params]

    def get_beta_values(self):
        return [p() for p in self.params]


class Expressions:
    def __init__(self):
        pass

    def __add__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return self.sharedVar + other
        return super().__add__(other)

    def __radd__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return self.sharedVar + other
        return super().__radd__(other)

    def __sub__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return self.sharedVar - other
        return super().__sub__(other)

    def __rsub__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return other - self.sharedVar
        return super().__rsub__(other)

    def __mul__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return self.sharedVar * other
        return super().__mul__(other)

    def __rmul__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return other * self.sharedVar
        return super().__rmul__(other)

    def __div__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return self.sharedVar / other
        return super().__div__(other)

    def __rdiv__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return other / self.sharedVar
        return super().__rdiv__(other)

    def __truediv__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return self.sharedVar / other
        return super().__truediv__(other)

    def __rtruediv__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return other / self.sharedVar
        return super().__rtruediv__(other)

    def __neg__(self):
        if isinstance(self, (TensorVariable, BetaShared)):
            return -self
        return super().__neg__()

    def __pow__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return self.sharedVar ** other
        return super().__pow__(other)

    def __rpow__(self, other):
        if isinstance(other, (TensorVariable, BetaShared)):
            return other ** self.sharedVar
        return super().__pow__(other)


class BetaShared(Expressions, Beta):
    def __init__(self, name, value, lowerbound, upperbound, status):
        Beta.__init__(self, name, value, lowerbound, upperbound, status)
        self.sharedVar = aesara.shared(value=np.array(value).astype(floatX), name=name)

    def __call__(self):
        return self.sharedVar

    def __str__(self):
        return f"{self.sharedVar.get_value()}"

    def __repr__(self):
        value = self.sharedVar.get_value()
        return (
            f"BetaShared('{self.name}', {value}, {self.lb}, {self.ub}, {self.status})"
        )


class DatabaseShared(Database):
    def __init__(self, name, pandasDatabase, choiceVar):
        super().__init__(name, pandasDatabase)

        for v in self.variables:
            if v in self.data.columns:
                if self.variables[v].name == choiceVar:
                    self.variables[v].y = aet.ivector(self.variables[v].name)
                else:
                    self.variables[v].x = aet.matrix(self.variables[v].name)

    def get_x(self):
        list_of_x = []
        for var in self.variables:
            if hasattr(self.variables[var], "x"):
                list_of_x.append(self.variables[var].x)
        return list_of_x

    def get_x_data(self, index=None, batch_size=None, shift=None):
        x_data = []
        list_of_x = self.get_x()
        for x in list_of_x:
            if index == None:
                x_data.append(self.data[[x.name]])
            else:
                start = index * batch_size + shift
                end = (index + 1) * batch_size + shift
                x_data.append(self.data[[x.name]][start:end])
        return x_data

    def get_y(self):
        list_of_y = []
        for var in self.variables:
            if hasattr(self.variables[var], "y"):
                list_of_y.append(self.variables[var].y)
        return list_of_y

    def get_y_data(self, index=None, batch_size=None, shift=None):
        y_data = []
        list_of_y = self.get_y()
        for y in list_of_y:
            if index == None:
                y_data.append(self.data[y.name])
            else:
                start = index * batch_size + shift
                end = (index + 1) * batch_size + shift
                y_data.append(self.data[y.name][start:end])
        return y_data

    def inputs(self):
        return self.get_x() + self.get_y()

    def input_data(self, index=None, batch_size=None, shift=None):
        """ Outputs a list of pandas table data corresponding to the 
            Symbolic variables

        Args:
            index (int, optional): Starting index slice.
            batch_size (int, optional): Size of slice.
            shift (int, optional): Add a random shift of the index.

        Returns:
            list: list of database (input) data
        """
        return self.get_x_data(index, batch_size, shift) + self.get_y_data(
            index, batch_size, shift
        )

    def autoscale(self, variables=None, verbose=False, default=None):
        for d in self.data:
            max_val = np.max(self.data[d])
            min_val = np.min(self.data[d])
            scale = 1.0
            if variables is None:
                varlist = self.get_x()
            else:
                varlist = variables
            if d in varlist:
                if min_val >= 0.0:
                    if default is None:
                        while max_val > 10:
                            self.data[d] /= 10.0
                            scale *= 10.0
                            max_val = np.max(self.data[d])
                    else:
                        self.data[d] /= default
                        scale = default
                if verbose:
                    print("scaling {} by {}".format(d, scale))


def build_functions(model, optimizer=None):
    lr = aet.scalar("learning_rate")
    if optimizer is not None:
        opt = optimizer(model.params)
        updates = opt.update(model.cost, model.params, lr)

        model.loglikelihood_estimation = aesara.function(
            inputs=model.inputs + [lr],
            outputs=model.cost,
            updates=updates,
            on_unused_input="ignore",
        )

    model.loglikelihood = aesara.function(
        inputs=model.inputs,
        outputs=full_loglikelihood(model.p_y_given_x, model.y),
        on_unused_input="ignore",
    )

    model.output_probabilities = aesara.function(
        inputs=model.inputs, outputs=model.p_y_given_x, on_unused_input="ignore",
    )

    model.output_estimated_betas = aesara.function(
        inputs=model.inputs, outputs=model.get_beta_values(), on_unused_input="ignore",
    )

    model.output_errors = aesara.function(
        inputs=model.inputs,
        outputs=errors(model.p_y_given_x, model.y),
        on_unused_input="ignore",
    )

    return model


def train(Model, db, optimizer, batch_size=256, max_epoch=2000, lr_init=0.01, seed=0):
    assert isinstance(Model, type)
    rng = np.random.default_rng(seed)

    print("Building model...")
    model = build_functions(Model(db), optimizer)

    n_samples = len(db.data)
    n_batches = n_samples // batch_size
    patience = 8000
    patience_increase = 2
    validation_threshold = 1.003
    validation_frequency = min(n_batches, patience / 2)
    done_looping = False
    early_stopping = False
    total_iter = max_epoch * n_batches
    epoch = 0

    print("dataset: {} ({})".format(db.name, n_samples))
    print("batch size: {}".format(batch_size))
    print("batches per epoch: {}".format(n_batches))
    print("validation frequency: {}\n".format(validation_frequency))
    print("Training model...")

    model.null_ll = model.loglikelihood(*db.input_data())
    model.best_ll = model.null_ll
    pbar0 = tqdm.tqdm(
        bar_format=("Loglikelihood:  {postfix[0][ll]:.6f}"),
        postfix=[dict(ll=model.null_ll)],
        position=0,
        leave=True,
    )
    pbar = tqdm.tqdm(
        total=total_iter,
        desc="Epoch {0:4d}/{1}".format(0, total_iter),
        unit_scale=True,
        position=1,
        leave=True,
    )

    while (epoch < max_epoch) and (not done_looping):
        epoch = epoch + 1

        for batch_index in range(n_batches):
            iter = (epoch - 1) * n_batches + batch_index

            lr = learn_rate_tempering(iter, patience, lr_init)
            i = rng.integers(0, n_batches)
            shift = rng.integers(0, batch_size)

            # train model
            model.loglikelihood_estimation(*db.input_data(i, batch_size, shift) + [lr])

            # validation step
            if (iter + 1) % validation_frequency == 0:
                ll = model.loglikelihood(*db.input_data())
                if ll > model.best_ll:
                    model.best_ll = ll
                    model.best_epoch = epoch
                    pbar0.postfix[0]["ll"] = model.best_ll
                    pbar0.update()

                    if ll > (model.best_ll * validation_threshold):
                        patience = max(patience, iter * patience_increase)
                        best_model = model

            pbar.set_description("Epoch {0:4d}/{1}".format(epoch, max_epoch),)
            pbar.update()

            if patience <= iter:
                done_looping = True
                early_stopping = True
                break

    model.best_error = best_model.output_errors(*db.input_data())
    with open("best_model.pkl", "wb") as f:
        pickle.dump(model, f)  # save model to pickle

    if early_stopping:
        print("Estimation reached maximum patience. Early stopping...")
    print(
        (
            "Optimization complete with best validation error of {0:6.3f}%\n"
            " with maximum loglikelihood reached @ epoch {1}."
        ).format(model.best_error * 100.0, model.best_epoch)
    )
    pbar0.close()
    pbar.close()
    return best_model

