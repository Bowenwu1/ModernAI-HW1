from scipy.optimize import leastsq
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import math

def cost(p, x, y, use_reg, lam):
    ret = y - np.poly1d(p)(x)
    if use_reg:
        reg = np.sqrt(math.exp(lam)) * (p * p)
        ret = np.append(ret, reg)
    return ret


def poly_fit(x, y, order, use_reg=False, lam=0):
    p = np.zeros((order + 1,1))
    coff = leastsq(cost, p, args=(x, y, use_reg, lam))
    curve_x = np.linspace(min(x), max(x), 1000)
    return curve_x, np.poly1d(coff[0])(curve_x)

# def poly_fit(x, y, order, lam=0):
#     model = make_pipeline(PolynomialFeatures(order), Ridge(alpha=math.exp(lam)))
#     model.fit(x.reshape(-1, 1), y)
#     curve_x = np.arange(min(x), max(x) + 0.01, 0.01)
#     curve_y = model.predict(curve_x.reshape(-1, 1))
#     return curve_x, curve_y