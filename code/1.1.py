import numpy as np; np.random.seed(300)
import argparse
import os
import os.path as osp
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import seaborn as sns; sns.set()
from plot_utils import draw_line
from plot_utils import draw_scatter
from plot_utils import x_y_to_pd
from poly_fit import poly_fit

parser = argparse.ArgumentParser()
parser.add_argument('--point_num', type=int, default=10)
parser.add_argument('--order', type=int, default=3)
parser.add_argument('--use_reg', action='store_true')
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--save_dir', type=str, default='../fig/curve_result/')
opt = parser.parse_args()

if opt.use_reg:
    result_dir = osp.join(opt.save_dir, str(opt.point_num) + '_' + str(opt.order) + '_reg_' + str(opt.lam))
else:
    result_dir = osp.join(opt.save_dir, str(opt.point_num) + '_' + str(opt.order))
if not osp.exists(result_dir):
    os.makedirs(result_dir)

# s(n) = sin(n)
n = np.linspace(0, 1, opt.point_num)
s_n = np.sin(2 * np.pi * n)

# plot s(n)
sns.lineplot(x='x', y='y', data=x_y_to_pd(n, s_n), color='green')
plt.title(r'$sin(n)$')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(osp.join(result_dir, 's_n.png'))
plt.close()


# x(n) = s(n) + w(n)
x_n = s_n + (np.random.rand(s_n.size) - 0.5)
# plot x(n)
sns.lineplot(x='x', y='y', data=x_y_to_pd(n, s_n), color='green')
kwargs = {'color':'blue'}
sns.scatterplot(x='x', y='y', data=x_y_to_pd(n, x_n), **kwargs)
plt.savefig(osp.join(result_dir, 'x_n.png'))
plt.close()

curve_x, curve_y = poly_fit(n, x_n, opt.order, opt.use_reg, opt.lam)
sns.lineplot(x='x', y='y', data=x_y_to_pd(curve_x, curve_y), color='red')
sns.lineplot(x='x', y='y', data=x_y_to_pd(n, s_n), color='green')
sns.scatterplot(x='x', y='y', data=x_y_to_pd(n, x_n), **kwargs)
if opt.use_reg:
    plt.title(r'$order={}$;$ln\lambda={}$'.format(opt.order, opt.lam))
else:
    plt.title(r'$order={}$'.format(opt.order))
plt.text(0.6, 0.8, r'$N={}$'.format(n.size))
plt.savefig(osp.join(result_dir, 'curve.png'))