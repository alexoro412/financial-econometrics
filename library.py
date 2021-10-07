from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt


def compute_percent_returns(data):
    """
    Compute the percent returns.
    NB: The index may be messed up, since a row gets dropped
    """
    # There are weird issues with rows 0 and 1, and NaN
    # This should work though
    percent_returns = data.diff()[1:] / data.shift(1)
    percent_returns = percent_returns.drop(0)
    return percent_returns


def show_acf(data, nlags=48, title=""):
    """
    Plot the autocorrelation function, and also print out a table
    """
    ac, qstats, pvalues = acf(data, nlags=nlags, fft=False, qstat=True)
    # TODO verify which pacf method is best
    pac = pacf(data, nlags=nlags)
    # Drop the autocorrelations with lag 0
    pac = pac[1:]
    ac = ac[1:]
    print("Autocorrelation Function")
    print(pd.DataFrame({"acf": ac, "pacf": pac, "Q": qstats, "p": pvalues}))
    plot_acf(data, zero=False, title=title,
             lags=nlags)
    plt.ylabel("ACF")
    plt.xlabel("# Lags")


def fit_ar_one_model(data):
    """
    Fit an AR(1) model to some data
    (may need to call .to_numpy() on pandas data)
    """
    ar_model = AutoReg(endog=data, lags=[1], old_names=True).fit()
    return ar_model

def fit_arma_model(data, p, q):
    """
    Fit an ARMA(p,q) model to some data
    """
    arma_model = ARIMA(endog=data, order=(p,0,q)).fit()
    return arma_model


def plot_prediction(df, column, ar_model):
    """
    Plot the prediction from an AR(1) model against
    the actual data
    """
    df["prediction"] = ar_model.predict(start=0, end=len(df), dynamic=False)
    df.plot(x="DATE", y=[column, "prediction"], ylabel="Growth rate")


def jarque_bera(data):
    """
    Run the Jarque-Bera test, I guess, whatever that means
    """
    jb_value, p_value = scipy.stats.jarque_bera(data)
    print("Jarque Bera Test")
    print("JB: ", jb_value)
    print("p: ", p_value)
