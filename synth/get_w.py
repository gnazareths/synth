from scipy.optimize import fmin_slsqp

def get_w(w, v, x0, x1):
    result = fmin_slsqp(w_rss, w, f_eqcons=w_constraint, bounds=[(0.0, 1.0)]*len(w),
             args=(v, x0, x1), disp=False, full_output=True)
    weights = result[0]
    return weights
