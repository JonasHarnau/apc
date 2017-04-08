import matplotlib.pyplot as plt
import pandas as pd
from .get_data_sums import get_data_sums

def plot_data_sums(formatted_data, data_type = 'all', average = False,
                   log_scale = False, title = None, figsize = None):

    """
    
    Plot for data sums or averages by age, period and cohort.
    
    Produces plots showing age, period and cohort sums. As a default this is done for 
    responses, doses and rates.
    
    
    Parameters
    ----------
    
    formatted_data : output of 'apc.format_data'
    
    data_type : {'response', 'dose', 'rate', 'mortality', 'all'}, optional
                Determines what the sums are computed for. 'mortlity' maps to 'rate'. If
                'all' is chosen and 'dose' is 'None', output is for 'response'.
                
    average : bool, optional
              Determines whether sums or averages are computed. (Default is 'False')
    
    log_scale : bool, optional
                Specifies whether the y-axis uses a log-scale.  (Default is 'False')
                
    title : str, optional
            Sets the canvas title. If 'None' this is set to "Sums of data by
            age/period/cohort" or "Averages of data by age/period/cohort" as
            appropriate.
            
    figsize : float tuple or list, optional
              Specifies the figure size. If left empty matplotlib determines this
              internally.
    
    
    Returns
    -------
    
    Matplotlib figure
    
    Examples
    --------
    
    >>> import apc
    >>> data = apc.data_Italian_bladder_cancer()
    >>> apc.plot_data_sums(data)
    
    >>> import apc
    >>> data = apc.data_asbestos()
    >>> apc.plot_data_sums(data, average = True)
    
    """
    
    data_sums = get_data_sums(formatted_data, data_type, average)
    
    if data_type.lower() == 'all' and formatted_data.dose is None:
        data_type = 'Response'
    
    fig, axes = plt.subplots(nrows= 3 if data_type.lower() == 'all' else 1, ncols = 3,
                             figsize = figsize, sharex = 'col')
    
    # Make axes into array so the code following thise works in general.
    if data_type.lower() != 'all':
        axes = axes.reshape((1,3))
    
    data_sums.by_age.plot(ax = axes[:,0], subplots = True, logy = log_scale, legend = False)
    data_sums.by_period.plot(ax = axes[:,1],subplots = True, logy = log_scale, legend =False)
    data_sums.by_cohort.plot(ax = axes[:,2],subplots = True, logy = log_scale, legend =False)
    
    sum_or_avg_label = 'Averages' if average else 'Sums'
        
    if title is None:
        title = '{} of data by age/period/cohort'.format(sum_or_avg_label)
    
    fig.canvas.set_window_title('{}'.format(title))
    
    axes[0,0].set_ylabel('{} {}'.format('Response' if data_type.lower() == 'all' else data_type.title(), 
                                        sum_or_avg_label[:-1].lower()))
    if data_type.lower() == 'all':
        axes[1,0].set_ylabel('Dose {}'.format(sum_or_avg_label[:-1].lower()))
        axes[2,0].set_ylabel('Rate {}'.format(sum_or_avg_label[:-1].lower()))
    
    fig.tight_layout()
        
    return fig
    