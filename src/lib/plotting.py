import itertools
from matplotlib import use as plot_backend_use
plot_backend_use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def training_metrics_plot(filepath,training_values, validation_values, metric='accuracy',filetype='png'):

    # Plot of the loss for training and validation after every epoch
    # --------------------------------------------------------------
    
    if metric == 'accuracy':
        plot_title = 'Training and validation accuracy values after each epoch'
        plot_ylabel = 'Avg Accuracy'
    elif metric == 'loss':
        plot_title = 'Training and validation loss values after each epoch'
        plot_ylabel = 'Avg Loss'

    plt.figure(figsize=(8, 6))

    # training loss
    x = [i+1 for i in range(0, len(training_values)) ]
    y = training_values
    training_plot = 'b^--'
    plt.plot(x, y, training_plot)

    # validation loss
    y = validation_values
    validation_plot = 'r.--'
    plt.plot(x, y, validation_plot)

    # legend, and labels
    plt.legend(['Training','Validation'])
    plt.title(plot_title)
    plt.xlabel('Epoch count')
    plt.ylabel(plot_ylabel)
    plt.xticks(x)

    filepath = '.'.join([filepath,filetype])
    plt.savefig(filepath)


def confusion_matrix(filepath,cm,classes,cmap,normalize=False,title='Confusion matrix',
                     font_family='Sans',font_properties=None,filetype='png'
                    ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if font_properties is None:
        #font_properties = [26,16,17,20,40,20]
        font_properties =  [12,14,14,16,14,40,12]

    title_font_size = font_properties[0]
    cell_font_size = font_properties[1] 
    tick_font_size = font_properties[2] # the class labels
    label_font_size = font_properties[3] # the axes
    cbar_font_size = font_properties[4]
    x_rotation = font_properties[5]
    cbar_pad = font_properties[6]
    family = font_family
    ha = 'right'
    
    if normalize:
        cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm_n.max() / 2.
    else:
        thresh = cm.max() / 2.
    
    fig = plt.figure(figsize=(10,10), dpi=400)
    ax1 = fig.add_subplot(1, 1, 1,  aspect='equal')
    
    if normalize:
        plt.imshow(cm_n, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        
    plt.title(title, size=title_font_size, family=family)
    
    # Main plot
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=x_rotation, size=tick_font_size, family=family,  ha=ha)
    plt.yticks(tick_marks, classes, size=tick_font_size, family=family)

    if normalize:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     size=cell_font_size,
                     color="white" if cm_n[i, j] > thresh else "black", 
                     family=family
                    )
    else:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     size=cell_font_size,
                     color="white" if cm[i, j] > thresh else "black",
                     family=family
                    )   

    plt.ylabel('True Label', size=label_font_size, family=family)
    plt.xlabel('Predicted Label', size=label_font_size, family=family)
    
    # Colorbar legend
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    cbar = plt.colorbar(cax=cax)
    cbar.ax.get_yaxis().labelpad = cbar_pad
    #cbar.ax.tick_params(family=family)
    cbar.set_label('Class Sample Proportion', rotation=270, size=cbar_font_size, family=family)

    plt.tight_layout()

    filepath = '.'.join([filepath,filetype])
    plt.savefig(filepath)