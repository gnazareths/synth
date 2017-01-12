import pandas as pd

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
        
    estimates, predictors, controls = get_estimate(X0, X1, Z0, Z1, Y0)
    
    estimated_predictors = np.dot(X0,controls)
    predictors_table = pd.DataFrame({'Synthetic':estimated_predictors, 'Actual': X1},index=X1.index)
    
    estimated_outcomes = np.dot(Y0,controls)
    outcomes_table = pd.DataFrame({'Synthetic':estimated_outcomes, 'Actual':Y1},index=Y1.index)
        
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
    
    return estimates, Y1, predictors_table, outcomes_table, predictors, controls
