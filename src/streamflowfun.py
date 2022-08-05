import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")


def load_usgs_gauge_data(filepath="data/Canning_usgs_daily_mean_discharge_clean.csv"):
    """
    load gauge data and make some new columns

    requires pandas
    Inputs:
    filepath = usgs gauge data csv directory
    Outputs:
    data = pandas df

    """
    # read the data file and fill nans with 0
    data = pd.read_csv(filepath).fillna(0)
    new_column_names = ["Agency", "SiteNo", "OldDateTime", "Discharge_cfs", "Errorflag"]
    data.columns = new_column_names
    # add some new columns
    data["Discharge_m3sec"] = data["Discharge_cfs"] * 0.028316847
    data["DateTime"] = pd.to_datetime(data["OldDateTime"])
    data["DOY"] = data["DateTime"].dt.dayofyear
    data["Month"] = data["DateTime"].dt.month
    data["Year"] = data["DateTime"].dt.year
    return data


def load_usgs_field_data(filepath="data/Canning_usgs_field_measurements_clean.csv"):
    """
    load field data and make some new columns
    requires pandas
    Inputs:
    filepath = usgs field data csv directory
    Outputs:
    data = pandas df
    """
    # read the data file
    columns = [
        "control_type_cd",
        "measurement_dt",
        "tz_cd",
        "gage_height_va",
        "discharge_va",
        "chan_discharge",
        "chan_width",
        "chan_area",
        "chan_velocity",
        "chan_stability",
        "chan_material",
    ]
    data = pd.read_csv(filepath, usecols=columns)
    # add some new columns
    data["Discharge_m3sec"] = data["chan_discharge"] * 0.028316847
    data["Width_m"] = data["chan_width"] * 0.3048
    data["Velocity_msec"] = data["chan_velocity"] * 0.3048
    data["Area_m2"] = data["chan_area"] * 0.3048 * 0.3048
    data["Depth_m"] = data["Area_m2"] / data["Width_m"]
    return data.fillna(0)


def WfromQ_fun(Q, c, exp):
    return c * (Q ** exp)


def DfromQ_fun(Q, c, exp):
    return c * (Q ** exp)


def width_discharge_fit(inputdf, maxQ):
    """
    get the coefficient and exponent for W = C*Q^exp,
    using field data for W and Q near the gauge
    and make a figure

    requires scipy and matplotlib
    Inputs:
    inputdf = pandas df with usgs field data
    maxQ = maximum discharge from gauge hydrograph (m3/s)
    Outputs:
    coeff = C
    exp  = exp

    """
    df = inputdf.dropna().copy()  # ignore columns with nans
    width = (
        df.Width_m.values
    )  # nondim by mean annual width, se we can apply to other nodes upstream
    discharge = df.Discharge_m3sec.values

    # ice obstruction categories
    clear = df.control_type_cd.values == "Clear"
    icecover = df.control_type_cd.values == "IceCover"
    iceshore = df.control_type_cd.values == "IceShore"

    # find best-fit C and exp for ice-free data points
    [C, exp], pcov = curve_fit(WfromQ_fun, discharge[clear], width[clear])

    fig, ax = plt.subplots(1, 1)
    Q = np.linspace(0, maxQ, 1000)
    W = np.asarray([WfromQ_fun(i, C, exp) for i in Q])

    ax.scatter(discharge[clear], width[clear], c="k", label="clear")
    ax.scatter(
        discharge[icecover], width[icecover], c="r", label="ice cover",
    )
    ax.scatter(
        discharge[iceshore], width[iceshore], c="m", label="ice shore",
    )
    ax.plot(
        Q,
        W,
        color="k",
        label="W = " + str(round(C, 2)) + "*Q$^{" + str(round(exp, 2)) + "}$",
    )
    ax.set_xlabel("Discharge (m3/sec)")
    ax.set_ylabel("Width (m)")
    plt.legend()
    plt.show()
    return C, exp


def depth_discharge_fit(inputdf, maxQ):
    """
    get the coefficient and exponent for the D = C*Q^exp,
    using field data of depth and discharge
    and make a figure

    requires matplotlib and pandas
    Inputs:
    inputdf = pandas df with usgs field data
    maxQ = maximum discharge from gauge hydrograph (m3/s)

    Outputs:
    coeff = C
    exp  = exp
    """
    df = inputdf.dropna().copy()  # ignore columns with nans
    discharge = df.Discharge_m3sec.values
    depth = df.Depth_m.values

    # ice obstruction categories
    clear = df.control_type_cd.values == "Clear"
    icecover = df.control_type_cd.values == "IceCover"
    iceshore = df.control_type_cd.values == "IceShore"
    # find best-fit C and exp for ice-free data points
    [C, exp], pcov = curve_fit(DfromQ_fun, discharge[clear], depth[clear])
    fig, ax = plt.subplots(1, 1)
    Q = np.linspace(0, maxQ, 1000)
    D = np.asarray([DfromQ_fun(i, C, exp) for i in Q])

    ax.scatter(discharge[clear], depth[clear], c="k", label="clear")
    ax.scatter(
        discharge[icecover], depth[icecover], c="r", label="ice cover",
    )
    ax.scatter(
        discharge[iceshore], depth[iceshore], c="m", label="ice shore",
    )
    ax.plot(
        Q,
        D,
        color="k",
        label="D = " + str(round(C, 2)) + "*Q$^{" + str(round(exp, 2)) + "}$",
    )
    ax.set_xlabel("Discharge (m^3) ")
    ax.set_ylabel("Depth (m)")
    plt.legend()
    plt.show()
    return C, exp


def stage_at_doy(df, coeff_d, exp_d):
    """
    translate usgs gauge data into width and depth arrays
    Inputs:
    df =

    coeff_d =
    exp_d =

    """
    year = df["Year"].values
    discharge = df["Discharge_m3sec"].values
    DOY = df["DOY"].values

    mediandepth = np.zeros(len(range(0, np.max(DOY))))
    stddevdepth = np.zeros(len(range(0, np.max(DOY))))
    mindepth = np.zeros(len(range(0, np.max(DOY))))
    maxdepth = np.zeros(len(range(0, np.max(DOY))))
    medianQ = np.zeros(len(range(0, np.max(DOY))))
    stddevQ = np.zeros(len(range(0, np.max(DOY))))
    minQ = np.zeros(len(range(0, np.max(DOY))))
    maxQ = np.zeros(len(range(0, np.max(DOY))))

    for i in range(0, np.max(DOY)):
        allQ = discharge[np.where(DOY == i)[0]]
        if len(allQ) == 0:
            allQ = np.asarray([0])
        medianQ[i] = np.median(allQ)
        stddevQ[i] = np.std(allQ)
        maxQ[i] = np.max(allQ)
        minQ[i] = np.min(allQ)

        mediandepth[i] = DfromQ_fun(np.median(allQ), coeff_d, exp_d)
        stddevdepth[i] = np.std([DfromQ_fun(Qi, coeff_d, exp_d) for Qi in allQ])
        mindepth[i] = DfromQ_fun(np.min(allQ), coeff_d, exp_d)
        maxdepth[i] = DfromQ_fun(np.max(allQ), coeff_d, exp_d)
    return mediandepth, mindepth, maxdepth, stddevdepth, medianQ, minQ, maxQ, stddevQ
