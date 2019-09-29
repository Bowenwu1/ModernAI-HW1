import numpy as np; np.random.seed(123)
import argparse
import os
import os.path as osp
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from plot_utils import draw_line
from plot_utils import draw_scatter
from plot_utils import x_y_to_pd

# objective function
# https://zh.wikipedia.org/wiki/%E5%87%A0%E4%BD%95%E4%B8%AD%E5%BF%83
def loss(x, y, centroid_x, centroid_y):
    l2_distance_square = (x - centroid_x) ** 2 + (y - centroid_y) ** 2
    return sum(l2_distance_square) / len(x)

# gradient function
def gradient(x, y, centroid_x, centroid_y):
    d_x = 2 * (centroid_x - x)
    d_y = 2 * (centroid_y - y)
    return sum(d_x) / len(x), sum(d_y) / len(y)

def main(opt):
    result_dir = osp.join(opt.save_dir, "_".join((str(opt.lr), opt.optimizer, str(opt.n), str(opt.max_iter), str(opt.bs))))
    if not osp.exists(result_dir):
        os.makedirs(result_dir)

    # generate n points
    x = np.random.rand(opt.n)
    y = np.random.rand(opt.n)
    gt_centroid_x = np.average(x)
    gt_centroid_y = np.average(y)

    # plot n points
    sns.scatterplot(x='x', y='y', data=x_y_to_pd(x, y), alpha=0.6)
    sns.scatterplot(x='x', y='y', data=x_y_to_pd([gt_centroid_x], [gt_centroid_y]), color='red')
    plt.savefig(osp.join(result_dir, 'points.png'))
    plt.close()

    # init value for centroid
    centroid_x, centroid_y = 1.0, 1.0
    # middle result
    middle_x = []
    middle_y = []
    loss_record = []

    middle_x.append(centroid_x); middle_y.append(centroid_y)
    for iter_index in range(opt.max_iter):
        if opt.optimizer == 'sgd':
            start_index = (iter_index * opt.bs) % opt.n
            sample_x, sample_y = x[start_index:start_index + opt.bs], y[start_index:start_index + opt.bs]
        elif opt.optimizer == 'gd':
            sample_x, sample_y = x, y
        loss_value = loss(sample_x, sample_y, centroid_x, centroid_y)
        g_x, g_y = gradient(sample_x, sample_y, centroid_x, centroid_y)
        # update centroid
        centroid_x, centroid_y = centroid_x - opt.lr * g_x, centroid_y - opt.lr * g_y
        # save result for visualization
        middle_x.append(centroid_x); middle_y.append(centroid_y)
        # print result
        print('iter/total: {}/{}; loss={:.3f}, (g_x, g_y)=({:.3f}, {:.3f}), (x, y)=({:.3f}, {:.3f})'.format(
                                                        iter_index,
                                                        opt.max_iter,
                                                        loss_value,
                                                        g_x, g_y,
                                                        centroid_x, centroid_y))
        loss_record.append(loss_value)

    # draw process of GD/SGD
    sns.scatterplot(x='x', y='y', data=x_y_to_pd(x, y), alpha=0.6)
    sns.scatterplot(x='x', y='y', data=x_y_to_pd([gt_centroid_x], [gt_centroid_y]), color='red')
    sns.lineplot(x='x', y='y', data=x_y_to_pd(middle_x, middle_y), color='green')
    sns.scatterplot(x='x', y='y', data=x_y_to_pd(middle_x, middle_y), color='green')
    if opt.optimizer == 'gd':
        plt.title('lr={},optimizer={},iter={}\n (x,y)=({},{})'.format(opt.lr, opt.optimizer, opt.max_iter, centroid_x, centroid_y))
    elif opt.optimizer == 'sgd':
        plt.title('lr={},optimizer={},bs={},iter={}\n (x,y)=({},{})'.format(opt.lr, opt.optimizer, opt.bs, opt.max_iter, centroid_x, centroid_y))
    plt.savefig(osp.join(result_dir, 'descent.png'))
    plt.close()
    # draw loss
    sns.lineplot(x='x', y='y', data=x_y_to_pd(range(len(loss_record)), loss_record))
    plt.title('loss')
    plt.savefig(osp.join(result_dir, 'loss.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='gd', help='gd|sgd')
    parser.add_argument('--bs', type=int, default=2000, help='batch size')
    parser.add_argument('--n', type=int, default=2000, help='num of points')
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='../fig/gd_result/')
    opt = parser.parse_args()

    main(opt)