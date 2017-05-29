import pandas as pd
import numpy as np
from scipy.optimize import fmin_slsqp, minimize
import matplotlib.pyplot as plt

def dataprep(foo, 
             predictors, 
             treated_unit, 
             control_units, 
             index_variable, 
             measured_variable,
             time_variable,
             predict_time, 
             optimize_time, 
             plot_time, 
             function="mean"):

    ## check datatypes of the inputs, and raise errors while they do not match expectation

    ## index_variable

    if not type(index_variable) in [int,str]:
        raise NameError("index_variable should be a string or an integer, depending on the header")
    #if not foo[index_variable].dtype in [int,float]:
    #    raise NameError("index_variable contains non-numerical data")
    
    ## takes the variable that indexes the pandas dataframe,
    ## most likely the numbers or names of the units,
    ## and sets the index of the dataframe to that column:

    foo.index = foo.loc[:,index_variable]

    ## creates a list with the header of the database, if present;
    ## if the header is not present, the result should be a list
    ## with a range of n, being n the number of columns:

    header = list(foo.columns.values)

    ## creates a list with the units (control, treatment, and others)

    units = set(foo.index)
    units = [x for x in units if str(x) != 'nan']

    ## creates a list with time units

    years_represented = set(foo[time_variable])
    years_represented = [x for x in years_represented if str(x) != 'nan']

    ## foo

    if not type(foo) == pd.core.frame.DataFrame:
        raise NameError("foo should be a pandas dataframe")

    ## predictors

    if not type(predictors) == list:
        raise NameError("predictors should be a list")
    if len(predictors) < 1:
        raise NameError("please specify at least one predictor")
    if len(predictors) != len(set(predictors)):
        raise NameError("predictors contains duplicates")
    for i in predictors:
        if i not in header:
            raise NameError("predictor not found in the dataframe")

    ## treated_unit

    if not type(treated_unit) in [int,str]:
        raise NameError("treated_unit should be an integer or string")
    if not treated_unit in units:
        raise NameError("treated_unit not found")

    ## control_units

    if not type(control_units) == list:
        raise NameError("control_units should be a list")
    if len(control_units) < 2:
        raise NameError("please specify at least 2 control units")
    if len(control_units) != len(set(control_units)):
        raise NameError("control_units contains duplicates")
    for i in control_units:
        if i not in units:
          raise NameError("control unit " + i + " not in the dataframe")
    if treated_unit in control_units:
        raise NameError("control_units cannot contain treated_unit")

    ## measured_variable

    if not type(measured_variable) in [int,str]:
        raise NameError("measured_variable should be a string or an integer, depending on its index")
    if not foo[measured_variable].dtype in [int,float]:
        raise NameError("measured_variable contains non-numerical data")
    if not measured_variable in header:
        raise NameError("measured_variable not found")
    if measured_variable in predictors:
        raise NameError("measured_variable cannot be in predictors")

    ## time_variable

    if not type(time_variable) in [int,str]:
        raise NameError("time_variable should be a string or integer, depending on its index")
    if not time_variable in header:
        raise NameError("time_variable not found")

    ## predict_time

    if not type(predict_time) == list:
        raise NameError("predict_time should be a list")
    if len(predict_time) < 1:
        raise NameError("predict_time should contain at least one entry")    
    for i in predict_time:
        if i not in years_represented:
            raise NameError("time period " + i + " not in the dataframe")

    ## optimize_time

    if not type(optimize_time) == list:
        raise NameError("optimize_time should be a list")
    if len(optimize_time) < 1:
        raise NameError("optimize_time should contain at least one entry")
    for i in optimize_time:
        if i not in years_represented:
            raise NameError("time period " + i + " not in the dataframe")

    ## plot_time

    if not type(plot_time) == list:
        raise NameError("plot_time should be a list")
    if len(plot_time) < 1:
        raise NameError("plot_time should contain at least one entry")
    for i in plot_time:
        if i not in years_represented:
            raise NameError("time period " + i + " not in the dataframe")

    ## check all time units

    all_times = plot_time + optimize_time + predict_time
    for i in all_times:
        if not isinstance(i,int) or isinstance(i,float):
            raise NameError("time units should be numeric")

    ## check if the function input matches expectation
    
    if not function in ["mean","median"]:
        raise NameError("function should be 'mean', or 'median'")

    ## more checks

    ## check if there is duplicated data (more than one instance for
    ## the same unit at the same year)

    for i in units:
        for j in years_represented:
            sub_df = foo.loc[i]
            sub_df = sub_df.loc[sub_df[time_variable]==j]
            if len(sub_df) > 1:
                raise NameError("dataframe contains duplicate data: more than one instance for the same unit and time_variable")

    ## sort foo by index_variable and time_variable

    foo = foo.sort_values([index_variable, time_variable], ascending=[1,1])

    ## at this point, the function already checked for validity
    ## of the predictors and predict_times. Now it should see if
    ## the X matrix (predictors, predict_times) is valid.

    X = foo.loc[foo[time_variable].isin(predict_time)]
    X0 = X.loc[control_units,([time_variable] + predictors)]
    X1 = X.loc[treated_unit,([time_variable] + predictors)]

    ## loop through entries in the X0 and X1, and check for
    ## missing data (when there is no instance of a unit in a given
    ## year), repeated data (when there are more than one instances of
    ## a unit in a given year), and non-numerical data.
    
    for i in control_units:
        for j in predict_time:
            for k in predictors:
                df = X0.loc[i]
                df = df[(df[time_variable]==j)][k].values
                if len(df) == 0:
                    raise NameError("dataframe has missing data")
                elif len(df) > 1:
                    raise NameError("dataframe has more than one instance of the outcome variable for the same control unit and year")
                else:
                    outcome = df[0]
                    if pd.isnull(outcome):
                        print k, j
                        raise NameError("dataframe has missing data")
                    if not (isinstance(outcome,float) or isinstance(outcome,int)):
                        raise NameError("dataframe contains non-numerical data")

    for i in predict_time:
        for j in predictors:
            df = X1[(X1[time_variable]==i)][j].values
            if len(df) == 0:
                raise NameError("dataframe has missing data")
            elif len(df) > 1:
                raise NameError("dataframe has more than one instance of the outcome variable for the same control unit and year")
            else:
                outcome = df[0]
                if pd.isnull(outcome):
                    raise NameError("dataframe has missing data")
                if not (isinstance(outcome,float) or isinstance(outcome,int)):
                    raise NameError("dataframe contains non-numerical data")
    
    ## remove the time column from X0 and X1, and store the time to use as index

    X0, X1 = X0[predictors], X1[predictors]

    ## prep the predictors' matrix (X)
    
    if len(predict_time) > 1:
        if function == "mean":
            X0 = X0.groupby(X0.index).mean()
            X1 = X1.groupby(X1.index).mean()
        elif function == "median":
            X0 = X0.groupby(X0.index).median()
            X1 = X1.groupby(X1.index).median()
        else:
            raise NameError("function must be either 'mean' or 'median'")

    X1 = X1.transpose()
    X0 = X0.transpose()

    ## create the Z0 and Z1 matrices

    Z = foo.loc[foo[time_variable].isin(optimize_time)]
    Z0 = Z.loc[control_units,[time_variable, measured_variable]]
    Z1 = Z.loc[treated_unit,[time_variable, measured_variable]]

    ## loop through entries in the Z0 and Z1, and check for
    ## missing data (when there is no instance of a unit in a given
    ## year), repeated data (when there are more than one instances of
    ## a unit in a given year), and non-numerical data.
    
    for i in control_units:
        for j in optimize_time:
            df = Z0.loc[i]
            df = df[(df[time_variable]==j)][measured_variable].values
            if len(df) == 0:
                raise NameError("dataframe has missing data")
            elif len(df) > 1:
                raise NameError("dataframe has more than one instance of the outcome variable for the same control unit and year")
            else:
                outcome = df[0]
                if pd.isnull(outcome):
                    raise NameError("dataframe has missing data")
                if not (isinstance(outcome,float) or isinstance(outcome,int)):
                    raise NameError("dataframe contains non-numerical data")

    for i in optimize_time:
        df = Z1[(Z1[time_variable]==i)][measured_variable].values
        if len(df) == 0:
            raise NameError("dataframe has missing data")
        elif len(df) > 1:
            raise NameError("dataframe has more than one instance of the outcome variable for the same control unit and year")
        else:
            outcome = df[0]
            if pd.isnull(outcome):
                raise NameError("dataframe has missing data")
            if not (isinstance(outcome,float) or isinstance(outcome,int)):
                raise NameError("dataframe contains non-numerical data")

    ## remove the time column from Z0 and Z1, and store the time to use as index

    TZ = Z1[time_variable].values
    Z0, Z1 = Z0[measured_variable], Z1[measured_variable]

    ## further prepare the matrices

    Z0.index = [Z0.groupby(level=0).cumcount(), Z0.index]
    Z0 = Z0.unstack()
    Z0.index = TZ
    Z1.index = [Z1.groupby(level=0).cumcount(), Z1.index]
    Z1 = Z1.unstack()
    Z1.index = TZ

    ## create the Y0 and Y1 matrices

    Y = foo.loc[foo[time_variable].isin(plot_time)]
    Y0 = Y.loc[control_units,[time_variable, measured_variable]]
    Y1 = Y.loc[treated_unit,[time_variable, measured_variable]]

    ## loop through entries in the Y0 and Y1, and check for
    ## missing data (when there is no instance of a unit in a given
    ## year), repeated data (when there are more than one instances of
    ## a unit in a given year), and non-numerical data.

    ## this can be done more efficiently; there is no need for double-counting
    ## some of the outcome variables that overlap between Z and Y matrices.

    for i in control_units:
        for j in plot_time:
            df = Y0.loc[i]
            df = df[(df[time_variable]==j)][measured_variable].values
            if len(df) == 0:
                raise NameError("dataframe has missing data")
            elif len(df) > 1:
                raise NameError("dataframe has more than one instance of the outcome variable for the same control unit and year")
            else:
                outcome = df[0]
                if pd.isnull(outcome):
                    raise NameError("dataframe has missing data")
                if not (isinstance(outcome,float) or isinstance(outcome,int)):
                    raise NameError("dataframe contains non-numerical data")

    for i in optimize_time:
        df = Y1[(Y1[time_variable]==i)][measured_variable].values
        if len(df) == 0:
            raise NameError("dataframe has missing data")
        elif len(df) > 1:
            raise NameError("dataframe has more than one instance of the outcome variable for the same control unit and year")
        else:
            outcome = df[0]
            if pd.isnull(outcome):
                raise NameError("dataframe has missing data")
            if not (isinstance(outcome,float) or isinstance(outcome,int)):
                raise NameError("dataframe contains non-numerical data")

    ## remove the time column from Z0 and Z1, and store the time to use as index

    TY = Y1[time_variable].values
    Y0, Y1 = Y0[measured_variable], Y1[measured_variable]

    ## further prepare the Y0 and Y1 matrices

    Y0.index = [Y0.groupby(level=0).cumcount(), Y0.index]
    Y0 = Y0.unstack()
    Y0.index = TY
    Y1.index = [Y1.groupby(level=0).cumcount(), Y1.index]
    Y1 = Y1.unstack()
    Y1.index = TY

    return X0.as_matrix(), X1.as_matrix(), Y0.as_matrix(), Y1.as_matrix(), Z0.as_matrix(), Z1.as_matrix()

def w_rss(w, v, x0, x1):
    k = len(x1)
    importance = np.zeros((k,k))
    np.fill_diagonal(importance, v)
    predictions = np.dot(x0,w)
    errors = x1 - predictions
    weighted_errors = np.dot(importance, errors)
    weighted_rss = sum(weighted_errors**2)[0]
    return weighted_rss

def w_constraint(w, v, x0, x1):
    return np.sum(w) - 1

def v_rss(w, z0, z1):
    predictions = np.dot(z0,w)
    errors = z1 - predictions
    rss = sum(errors ** 2)[0]
    return rss

def get_w(w, v, x0, x1):
    weights = fmin_slsqp(w_rss, w, f_eqcons=w_constraint, bounds=[(0.0, 1.0)]*len(w),
             args=(v, x0, x1), disp=False, full_output=True)[0]
    return weights

def get_v_0(v, w, x0, x1, z0, z1):
    weights = get_w(w,v,x0,x1)
    rss = v_rss(weights, z0, z1)
    return rss

def get_v_1(v, w, x0, x1, z0, z1):
    result = minimize(get_v_0, v, args=(w, x0, x1, z0, z1), bounds=[(0.0, 1.0)]*len(v)).x
    #result = fmin_slsqp(get_v_0, v, args=(w, x0, x1, z0, z1), bounds=[(0.0, 1.0)]*len(v))
    return result

def get_estimate(x0, x1, z0, z1, y0):
    (k,j) = x0.shape
    v = [1.0]*k
    w = np.array([1.0/j]*j).transpose()
    predictors = get_v_1(v, w, x0, x1, z0, z1)
    controls = np.array(get_w(w, predictors, x0, x1)).transpose()
    estimates = np.dot(y0,controls)
    return estimates, predictors, controls

def synth_tables(foo, 
                 predictors, 
                 treated_unit, 
                 control_units, 
                 index_variable, 
                 measured_variable,
                 time_variable,
                 predict_time, 
                 optimize_time, 
                 plot_time,
                 function="mean"):

    control_units.sort()
    predictors.sort()
    plot_time.sort()

    X0, X1, Y0, Y1, Z0, Z1 = dataprep(foo, 
                 predictors, 
                 treated_unit, 
                 control_units, 
                 index_variable, 
                 measured_variable,
                 time_variable,
                 predict_time, 
                 optimize_time, 
                 plot_time, 
                 function="mean")

    (est, predict, ctrls) = get_estimate(X0, X1, Z0, Z1, Y0)
    predict = [ round(elem,4) for elem in predict ]
    ctrls = [ round(elem,4) for elem in ctrls ]
    estimated_predictors = np.dot(X0,ctrls).transpose()
    predictors_table = pd.DataFrame({'Synthetic':estimated_predictors, 'Actual': X1.transpose()[0]}, index=predictors)
    estimated_outcomes = np.dot(Y0,ctrls)
    outcomes_table = pd.DataFrame({'Synthetic':estimated_outcomes, 'Actual':Y1.transpose()[0]},index=plot_time)
    predictors_weights = pd.DataFrame({'Weight':predict}, index=predictors)
    controls_weights = pd.DataFrame({'Weight':ctrls}, index=control_units)

    print "Predictors Table"
    print "---"
    print predictors_table
    print " "
    print "Outcomes Table"
    print "---"
    print outcomes_table
    print " "
    print "Predictors' Weights"
    print "---"
    print predictors_weights
    print " "
    print "Controls' Weights"
    print "---"
    print controls_weights

    plt.plot(plot_time, est, plot_time, Y1)
    plt.xlim(plot_time[0],plot_time[-1])
    plt.ylim([0,.4])
    plt.show()

    return
    
## test code example

## some_dataframe = pd.read_excel("filename.xlsx", header=0)
## control_units = list(set(some_dataframe["region_name"]))
## control_units.sort()
## control_units = control_units[1:4] + control_units[6:]
## 
## predictors = [  "medical_legalized", "no_high_school", "high_school", "college",
##                 "hispanic_pop", "white_pop",
##                 "african_american_pop", "native_american_pop", "unemployment_rate",
##                 "urban_pop"
##              ]
##     
## synth_tables(  some_dataframe,
##                predictors,
##                "California",
##                control_units,
##                "region_name",
##                "marijuana_18_25",
##                "year",
##                [2000,2010],
##                range(2009,2015),
##                range(2009,2015)
##                )
