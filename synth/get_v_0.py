def get_v_0(v, w, x0, x1, z0, z1):
    weights = fmin_slsqp(w_rss, w, f_eqcons=w_constraint, bounds=[(0.0, 1.0)]*len(w),
             args=(v, x0, x1), disp=False, full_output=True)[0]
    rss = v_rss(weights, z0, z1)
    return rss
