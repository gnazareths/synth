import numpy as np
import pandas as pd
from scipy.optimize import fmin_slsqp, minimize
from matplotlib import pyplot as plt

def basic_dataprep(predictors_matrix, outcomes_matrix, 
             treated_unit, control_units, predictors_optimize, 
             outcomes_optimize, years_plot):
    # check if data types match expectations.
    if not type(predictors_matrix) == pd.core.frame.DataFrame:
        raise NameError("Error 1")
    elif not type(outcomes_matrix) == pd.core.frame.DataFrame:
        raise NameError("Error 2")
    elif not type(treated_unit) == str:
        raise NameError("Error 3")
    elif not type(control_units) == list:
        raise NameError("Error 4")
    elif not type(predictors_optimize) == list:
        raise NameError("Error 5")
    elif not type(outcomes_optimize) == list:
        raise NameError("Error 6")
    elif not type(years_plot) == list:
        raise NameError("Error 7")
    
    # if the list of controls contains the treated unit, remove treated unit.
    while treated_unit in control_units:
        control_units.remove(treated_unit)
    
    # check for empty lists
    if len(control_units) == 0 or len(predictors_optimize) == 0 or len(outcomes_optimize) == 0:
           raise NameError("Error 8")
    
    # check for whether there are repeated control units, or more controls
    # than columns in the input matrices.  
    if len(control_units) >= predictors_matrix.shape[1] or len(control_units) >= outcomes_matrix.shape[1]:
           raise NameError("Error 9")

    X1 = predictors_matrix[treated_unit]
    del predictors_matrix[treated_unit]
    X0 = predictors_matrix
    
    Z3 = outcomes_matrix.loc[years_plot][treated_unit]
    Z2 = outcomes_matrix.loc[years_plot][control_units]
    Z1 = outcomes_matrix.loc[outcomes_optimize][treated_unit]
    Z0 = outcomes_matrix.loc[outcomes_optimize][control_units]
             
    return X0, X1, Z0, Z1, Z2, Z3

def w_rss(w, v, x0, x1):
    k = len(x1)
    importance = np.zeros((k,k))
    np.fill_diagonal(importance, v)
    predictions = np.dot(x0, w)
    errors = x1 - predictions
    weighted_errors = np.dot(errors.transpose(), importance)
    weighted_rss = np.dot(weighted_errors,errors).item(0)
    return weighted_rss
    
def v_rss(w, z0, z1):
    predictions = np.dot(z0,w)
    errors = z1 - predictions
    rss = sum(errors**2)
    return rss

def w_constraint(w, v, x0, x1):
    return np.sum(w) - 1
    
def get_w(w, v, x0, x1):
    result = fmin_slsqp(w_rss, w, f_eqcons=w_constraint, bounds=[(0.0, 1.0)]*len(w),
             args=(v, x0, x1), disp=False, full_output=True)
    weights = result[0]
    return weights

def get_v_0(v, w, x0, x1, z0, z1):
    weights = fmin_slsqp(w_rss, w, f_eqcons=w_constraint, bounds=[(0.0, 1.0)]*len(w),
             args=(v, x0, x1), disp=False, full_output=True)[0]
    rss = v_rss(weights, z0, z1)
    return rss
    
def get_v_1(v, w, x0, x1, z0, z1):
    result = minimize(get_v_0, v, args=(w, x0, x1, z0, z1), bounds=[(0.0, 1.0)]*len(v))
    importance = result.x
    return importance
    
def get_estimate(x0, x1, z0, z1, z2):
    k,j = len(x0),len(x1)
    v = [1.0/k]*k
    w = np.array([1.0/j]*j)
    predictors = get_v_1(v, w, x0, x1, z0, z1)
    controls = get_w(w, predictors, x0, x1)
    z_estimates = np.dot(z2,controls)
    return z_estimates, predictors, controls

def synth_tables(predictors_matrix, outcomes_matrix, treated_unit, control_units, 
    predictors_optimize, outcomes_optimize, years_plot):
    
    X0, X1, Z0, Z1, Z2, Z3 = basic_dataprep(predictors_matrix, outcomes_matrix, 
        treated_unit, control_units, predictors_optimize, outcomes_optimize, 
        years_plot)
        
    estimates, predictors, controls = get_estimate(X0, X1, Z0, Z1, Z2)
    
    estimated_predictors = np.dot(X0,controls)
    predictors_table = pd.DataFrame({'Synthetic':estimated_predictors, 'Actual': X1},index=X1.index)
    
    estimated_outcomes = np.dot(Z2,controls)
    outcomes_table = pd.DataFrame({'Synthetic':estimated_outcomes, 'Actual':Z3},index=Z3.index)
        
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
    print predictors
    print " "
    print "Controls' Weights"
    print "---"
    print controls  
    
    return estimates, Z3, predictors_table, outcomes_table, predictors, controls
    
