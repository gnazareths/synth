# Synth

## Introduction

A Python package for implementing the Synthetic Control Method for comparative case studies.

The Synthetic Control Method has been used in studies estimating effect of an intervention when a singly unit is exposed to it. Jens Hainmueller, the maintainer of Synth’s R package, here describes the method as a “procedure to construct synthetic control units based on a convex combination of comparison units that approximates the characteristics of the unit that is exposed to the intervention.”

Before continuing, I'd like to thank Alexis Diamond for the amazing feedback. I'd also like to stress that I am basing my algorithms on the calculations seen in the papers mentioned in the references below. The works of Abadie, Diamond and Hainmueler, thus, were crucial for me to develop this implementation.

Furthermore, I stress that this is a **simplification** of Hainmueller's implementations of Synth in R and MATLAB. I go over how I simplified the algorithms in the section named "Current limitations and future versions".

## How Synth works

This version of Synth takes as inputs two matrices: outcomes_matrix and predictors_matrix. The former matrix is exactly on what a researcher would like to measure effect. Conversely, the latter matrix is comprised of data that can effectively predict the outcomes for a unit.

To illustrate this, allow me to cite Abadie, Diamond, and Hainmueller (2010), who estimate the effects of California’s Proposition 99 (tobacco control program) in the per capita consumption of cigarettes. In this study, the outcomes_matrix would have data on the per capita consumption of cigarettes for unit and every year. If the study looks at t time periods, a treated unit, and j control units, the outcomes_matrix would have shape (t, j+ 1). Meanwhile, predictors_matrix is comprised of variables that can predict a unit’s outcomes. For example, the authors of the paper included Log(GDP per capita), retail price of cigarettes, and consumption of beer, among others. If the study looks at k predictors, the predictors_matrix would have shape (k, j+1).

The Synthetic Control Method uses both matrices to find 1) how important each predictor is in determining the outcome, and 2) how to construct a synthetic unit that best represents the treated one in terms of control units. For representing the importance of each predictor, Synth constructs a diagonal matrix V of length k by k, in which the term V.item(i,i) is the importance of the ith predictor for i in [0, 1, ..., k - 1]. For representing the weight of each control unit in the construction of the optimal synthetic one, Synth creates a vector W of length j, in which W[i] is the weight of the ith control unit for i in [0, 1, ..., j - 1].

To find the two unknowns, matrix V and vector W.

W* is the set of weights that minimizes (X0 W - X1)'V(X0  W - X1), in which X0 is the matrix of predictors for control units, X1is the vector of predictors for the treated unit, and V is the matrix with the importance of each predictor. As one can see, W* is a function of V: different matrices V yield different optimal weights. On the other hand, V* minimizes (Z0 W*(V) - Z1)'(Z0  W*(V) - Z1), in which Z0 is the matrix of outcomes for control units, Z1is the matrix of outcomes for the treated unit, and W*(V) is the optimal combination of weights given the matrix V.

In this implementation, Synth uses a nested optimization with SciPy quadratic programming minimization functions optimize.minimize and optimize.fmin_slsqp. To achieve this, I call my function get_v_1, which finds the variable v that minimizes the function get_v_0, which in turn calls the function get_w and returns the residual sum of squares associated with given values of v and w. This last function, get_w, finds the variable w that minimizes the rss associated with v and any values of w.

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

Upon finding the optimal vector W (and, by definition, V as well), Synth multiplies the Z0 (outcomes of the control units) by W to find an estimate (or a counterfactual) of the Z1 vector. Synth can plot both the estimate Z1 and the actual Z1, so the researcher can compare how well the model represents the actual data.

## Why Synth is awesome (or dope, depending on your location)

The breakthrough here is that Synth often returns an extremely well-fitted model of the actual data during the periods for which it optimizes. Imagine this scenario: you want to study the effects of an event that happened in 1999 by plotting the outcomes of the synthetic controls against the actual outcomes between 1990 and 2010. It only makes sense to optimize Synth for the periods [1990:1998]. This way, you will see how much the actual data will deviate from the synthetic once, which was calibrated with outcomes and predictors before the event.

## How to call Synth


    output = synth_tables( predictors_matrix,   ## pandas dataframe
                           outcomes_matrix,     ## pandas dataframe
                           treated_unit,        ## string with the index of the treated unit
                           control_units,       ## array with indexes all control units
                           predictors_optimize, ## to be clear, these are the predictors that matter, 
                                                ## in case there are extra ones.
                           outcomes_optimize,   ## these are the time periods for which you are optimizing the RSS.
                                                ## you should not include anything after the treatment to ensure validity.
                           years_plot           ## years which you want to plot
                     )
    output

    def plot(synth_tables):
        estimates, actual_values = synth_tables[0], synth_tables[1]
        plt.plot(range(len(estimates)),estimates, 'r--', label="Synthetic Control")
        plt.plot(range(len(estimates)),actual_values, 'b-', label="Actual Data")
        plt.title("Example Synthetic Control Model")
        plt.ylabel("Y axis")
        plt.xlabel("X axis")
        plt.legend(loc='upper left')
        plt.show()

    plot(output)
    
## Test code

#### 1) Regular scenario: returns a fairly well-fitted model and strong deviation after treatment

    pred = pd.DataFrame({'a':[3,1,6,1], 'b':[4,2,5,0], 'c':[5,2,3,5],
                             'd':[4,2,4,2], 'e': [3,4,7,2]},index=['A','B','C','D'])

    outc = pd.DataFrame({'a':[11,10,12,13,13,12,13], 'b':[8,8,10,11,11,11,12],
                             'c':[20,21,25,27,27,28,29], 'd':[16,17,22,25,25,26,27],
                             'e':[14,14,17,20,21,21,23]}, 
                             index=[2010,2011,2012,2013,2014,2015,2016])

    output = synth_tables( pred,
                           outc,
                           'a',
                           ['b','c','d','e'],
                           ['A','B','C','D','E'],
                           [2010,2011,2012,2013,2014],[2010,2011,2012,2013,2014,2015,2016]
                         )
    output

    >>>
    
    Predictors Table
    ---
       Actual  Synthetic
    A       3   3.728477
    B       1   2.543047
    C       6   5.543047
    D       1   0.543047

    Outcomes Table
    ---
          Actual  Synthetic
    2010      11   9.629141
    2011      10   9.629141
    2012      12  11.900664
    2013      13  13.443711
    2014      13  13.715234
    2015      12  13.715234
    2016      13  14.986758

    Predictors' Weights
    ---
    [ 0.24979485  0.21519529  0.35312926  0.17443384]

    Controls' Weights
    ---
    [  7.28476564e-01   2.77555756e-17  -5.55111512e-17   2.71523436e-01]
    
#### 2) Vector X1 is equal to one of the columns in X0: predictors are meaningless and errors are very large

    pred = pd.DataFrame({'a':[4,2,5,0], 'b':[4,2,5,0], 'c':[5,2,3,5],
                             'd':[4,2,4,2], 'e': [3,4,7,2]},index=['A','B','C','D'])

    outc = pd.DataFrame({'a':[11,10,12,13,13,12,13], 'b':[8,8,10,11,11,11,12],
                             'c':[20,21,25,27,27,28,29], 'd':[16,17,22,25,25,26,27],
                             'e':[14,14,17,20,21,21,23]}, 
                             index=[2010,2011,2012,2013,2014,2015,2016])

    output = synth_tables( pred,
                           outc,
                           'a',
                           ['b','c','d','e'],
                           ['A','B','C','D','E'],
                           [2010,2011,2012,2013,2014],[2010,2011,2012,2013,2014,2015,2016]
                         )
    output

    >>>
    Predictors Table
    ---
       Actual     Synthetic
    A       4  4.000000e+00
    B       2  2.000000e+00
    C       5  5.000000e+00
    D       0  6.317169e-14

    Outcomes Table
    ---
          Actual  Synthetic
    2010      11          8
    2011      10          8
    2012      12         10
    2013      13         11
    2014      13         11
    2015      12         11
    2016      13         12

    Predictors' Weights
    ---
    [ 0.25  0.25  0.25  0.25] ## THIS REPRESENTS THAT PREDICTORS ARE MEANINGLESS. This is the first
                              ## guess passed by the algorithm.

    Controls' Weights
    ---
    [  1.00000000e+00   1.53210777e-14  -4.66293670e-15  -2.05391260e-15]

#### 3) Treated unit's outcomes are equal to those of a control unit: model is perfectly well-fitted, because weight is always 1 for the equal unit and 0 for all others.

    pred = pd.DataFrame({'a':[3,1,6,1], 'b':[4,2,5,0], 'c':[5,2,3,5],
                             'd':[4,2,4,2], 'e': [3,4,7,2]},index=['A','B','C','D'])

    outc = pd.DataFrame({'a':[11,10,12,13,13,12,13], 'b':[11,10,12,13,13,12,13],
                             'c':[20,21,25,27,27,28,29], 'd':[16,17,22,25,25,26,27],
                             'e':[14,14,17,20,21,21,23]}, 
                             index=[2010,2011,2012,2013,2014,2015,2016])

    output = synth_tables( pred,
                           outc,
                           'a',
                           ['b','c','d','e'],
                           ['A','B','C','D','E'],
                           [2010,2011,2012,2013,2014],[2010,2011,2012,2013,2014,2015,2016]
                         )
    output

    >>>

    Predictors Table
    ---
       Actual     Synthetic
    A       3  4.000000e+00
    B       1  2.000000e+00
    C       6  5.000000e+00
    D       1 -7.667478e-15

    Outcomes Table
    ---
          Actual  Synthetic
    2010      11         11
    2011      10         10
    2012      12         12
    2013      13         13
    2014      13         13
    2015      12         12
    2016      13         13

    Predictors' Weights
    ---
    [ 0.10755938  0.49973705  0.31403663  0.        ]

    Controls' Weights
    ---
    [  1.00000000e+00  -2.24126273e-15   1.24900090e-16   1.64451786e-15]
    
#### 4) Treated unit's outcomes are equal to those of a control unit, but only for the period over which it's optimized: this is a perfect synthetic control, because the model will deviate a lot from actual data after the period for which it stops being optimized. In other words, 100% of the deviation would be due to the intervention.

    pred = pd.DataFrame({'a':[3,1,6,1], 'b':[4,2,5,0], 'c':[5,2,3,5],
                             'd':[4,2,4,2], 'e': [3,4,7,2]},index=['A','B','C','D'])

    outc = pd.DataFrame({'a':[11,10,12,13,13,15,17], 'b':[11,10,12,13,13,12,13],
                             'c':[20,21,25,27,27,28,29], 'd':[16,17,22,25,25,26,27],
                             'e':[14,14,17,20,21,21,23]}, 
                             index=[2010,2011,2012,2013,2014,2015,2016])

    output = synth_tables( pred,
                           outc,
                           'a',
                           ['b','c','d','e'],
                           ['A','B','C','D','E'],
                           [2010,2011,2012,2013,2014],[2010,2011,2012,2013,2014,2015,2016]
                         )
    output

    >>>

    Predictors Table
    ---
       Actual     Synthetic
    A       3  4.000000e+00
    B       1  2.000000e+00
    C       6  5.000000e+00
    D       1 -7.667478e-15

    Outcomes Table
    ---
          Actual  Synthetic
    2010      11         11
    2011      10         10
    2012      12         12
    2013      13         13
    2014      13         13
    2015      15         12
    2016      17         13

    Predictors' Weights
    ---
    [ 0.10755938  0.49973705  0.31403663  0.        ]

    Controls' Weights
    ---
    [  1.00000000e+00  -2.24126273e-15   1.24900090e-16   1.64451786e-15]

## Current limitations and future versions

Among the limitations of this implementation of Synth are:

1) Users cannot input their own V matrices

2) Users must input panda dataframes, rather than excel or csv files

3) Users cannot input special predictors, which are a part of Synth's R and MATLAB implementations.

I will focus on improving these in future versions of this code. I 100% welcome feedback from users.

## References: if you want to know more about Synth

[Package Synth (Hainmueller and Diamond, 2015)](https://cran.r-project.org/web/packages/Synth/Synth.pdf)

[Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California’s Tobacco Control Program (Abadie, Diamond and Hainmueller, 2012)](http://web.stanford.edu/~jhain/Paper/JASA2010.pdf)

[The Economic Costs of Conflict: A Case Study of the Basque Country (Abadie and Gardeazabal, 2003)](https://www.aeaweb.org/articles?id=10.1257/000282803321455188)
