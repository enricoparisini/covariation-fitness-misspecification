import numpy as np
import matplotlib.pyplot as plt

#Function that loads the results of a performance analysis and makes them easy to plot
def extract_performance(performance_filename):
    results = np.load(performance_filename, allow_pickle = True).item()
    labels = results["index"] 
    del(results["index"])

    N_effs = np.array([]).reshape(0,2)
    pearson_rs = np.array([]).reshape(0,len(labels)+1)
    spearman_rs = np.array([]).reshape(0,len(labels)+1)

    for key, value in results.items():
        N_effs = np.vstack((N_effs, [int(key), value[0]]))
        pearson_rs = np.vstack((pearson_rs, np.concatenate(([int(key)], value[1].T[0]))))
        spearman_rs = np.vstack((spearman_rs, np.concatenate(([int(key)], value[1].T[1]))))

    N_effs = N_effs.T
    pearson_rs = pearson_rs.T
    spearman_rs = spearman_rs.T
    return [N_effs, pearson_rs, spearman_rs, labels]


#Plot helper function that plots (on two columns) the performance of a model at different stages of training wrt different metrics.
def visualise_performance_single(performance_filename):
    N_effs, pearson_rs, spearman_rs, labels = extract_performance(performance_filename)
    n_y = 2
    n_x = 3
    scaling = 3.5
    _, axis = plt.subplots(n_x, n_y, sharex=True, figsize=(scaling*n_x, scaling*n_y)) 

    i_plot = 0
    for row in axis:
        for ax in row:
            if i_plot==0:
                ax.plot(N_effs[0], N_effs[1])
                ax.set_title("N_eff")
            elif i_plot==1:
                pass
            else:
                ax.plot(pearson_rs[0], pearson_rs[i_plot-1])
                ax.set_title(labels[i_plot-2])            
            i_plot+=1
    plt.subplots_adjust(hspace=scaling*0.12,wspace=scaling*0.07)
    plt.show() 


#Plot helper function that plots (on two columns) the performance of two models (or one model vs two different sets, 
#e.g. training and test sets) at different stages of training wrt different metrics.
def visualise_performance_double(performance_filename, performance_filename_2):
    N_effs, pearson_rs, spearman_rs, labels = extract_performance(performance_filename)
    N_effs_2, pearson_rs_2, spearman_rs_2, _ = extract_performance(performance_filename_2)
    n_y = 2
    n_x = 3
    scaling = 3.5
    _, axis = plt.subplots(n_x, n_y, sharex=True, figsize=(scaling*n_x, scaling*n_y)) 

    i_plot = 0
    for row in axis:
        for ax in row:
            if i_plot==0:
                ax.plot(N_effs[0], N_effs[1])
                ax.plot(N_effs_2[0], N_effs_2[1])
                ax.set_title("N_eff")
            elif i_plot==1:
                pass
            else:
                ax.plot(pearson_rs[0], pearson_rs[i_plot-1])
                ax.plot(pearson_rs_2[0], pearson_rs_2[i_plot-1])
                ax.set_title(labels[i_plot-2])            
            i_plot+=1
    plt.subplots_adjust(hspace=scaling*0.12,wspace=scaling*0.07)
    plt.show() 


def visualise_performance_train_test_1column(performance_filename, performance_filename_2, reduced_index, 
                                             filename_out = None, title = None, x_title = None,
                                             training_vs_test = None):
    '''
    Plot helper function that plots (on one column) the performance of two models (or one model vs two different sets, 
    e.g. training and test sets) at different stages of training wrt different metrics. The plot can be saved to file.
    Parameters:
    - performance_filename: (strings) path to the first performance file.
    - performance_filename_2: (strings) path to the second performance file.
    - reduced_index: (list) metrics to be plotted besides N_eff.
    - filename_out: (string) location of the file where the plot is written to.
    - title: (string) title of the figure.
    - x_title: (float) defines the horizontal position of the title. None corresponds to a centered title.
    '''
    N_effs, pearson_rs, spearman_rs, _ = extract_performance(performance_filename)
    _, pearson_rs_2, spearman_rs_2, _ = extract_performance(performance_filename_2)
    n_y = 1
    n_x = len(reduced_index)+1
    scaling = 20
    fig, axis = plt.subplots(n_x, n_y, sharex=True, figsize=(scaling/n_x, 0.6*scaling/n_y)) 

    i_plot = 0
    for row in axis:
        if i_plot==0:
            row.plot(N_effs[0], N_effs[1])
            row.set_title("N_eff")
            row.set_ylim(ymin=0, ymax=1.05*np.max(N_effs[1]))
        else:
            row.plot(pearson_rs[0], pearson_rs[i_plot])
            row.plot(pearson_rs_2[0], pearson_rs_2[i_plot])
            if training_vs_test is not None:
                x = np.array([np.min(pearson_rs[0]), np.max(pearson_rs[0])])
                y = np.array([training_vs_test[i_plot-1], training_vs_test[i_plot-1]])
                row.plot(x, y, color='black', linestyle='dashed')
            row.set_title("r_" + reduced_index[i_plot-1])            
        i_plot+=1
    if title is not None:
        fig.suptitle(title, fontsize=16, y = 0.95, x = x_title)
    plt.subplots_adjust(hspace=0.25,wspace=0.2)

    if filename_out is not None:
        plt.savefig(filename_out)  
    plt.show() 