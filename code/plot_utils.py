import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

def x_y_to_pd(x,y):
    import pandas as pd
    zippedList = list(zip(x,y))
    return pd.DataFrame(zippedList, columns=['x','y'])

def draw_scatter(x, y, save_path, x_label='x', y_label='y', whether_reg=True):
    ax = sns.scatterplot(x="x", y="y", data=x_y_to_pd(x,y))
    if whether_reg:
        ax = sns.regplot(x='x', y='y', data=x_y_to_pd(x,y), ci=None)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.rcParams["axes.labelsize"] = 22
    plt.tight_layout()
    # plt.figure(figsize=(8,4))
    print(save_path)
    plt.savefig(save_path)
    plt.close()

def draw_point(x, y, save_path, x_label='K', y_label='Metric'):
    sns.pointplot(x='x', y='y', data=x_y_to_pd(x, y))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_path)
    plt.close()

def draw_line(x, y, save_path, x_label='x', y_label='y', title=r'$sin(n)$'):
    sns.lineplot(x='x', y='y', data=x_y_to_pd(x, y))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()