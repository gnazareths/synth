def get_estimate(x0, x1, z0, z1, z2):
    k,j = len(x0),len(x1)
    v = [1.0/k]*k
    w = np.array([1.0/j]*j)
    predictors = get_v_1(v, w, x0, x1, z0, z1)
    controls = get_w(w, predictors, x0, x1)
    z_estimates = np.dot(z2,controls)
    return z_estimates, predictors, controls
