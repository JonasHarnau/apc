from .plot_data_heatmap import plot_data_heatmap
from .plot_data_sums import plot_data_sums
from .plot_data_within import plot_data_within

def plot_data_all(formatted_data, log_scale = False, average = False, figsize = None):
    
    """
    
    Makes all available data plots.
    
    This produces heatmap, sparsity, one time scale data sum and two time scale plots.
    
    
    Parameters
    ----------
    
    formatted_data : output of 'apc.format_data'
    
    average : bool, optional
              This argument passes through to 'plot_data_sums' and 'plot_data_within'.
              (Default is 'False')
    
    log_scale : bool, optional
                This argument passes through to 'plot_data_sums' and 'plot_data_within'.
                (Default is 'False')
    
    
    figsize : float tuple or list, optional
              Specifies the figure size. If left empty matplotlib determines this
              internally.    
    
    
    Returns
    -------
    
    Tuple of plots in the order
        heatmap, sparsity, sums, response_within, dose_within, rate_within
    
    """
    
    heatmap = plot_data_heatmap(formatted_data, figsize = figsize)
    
    sparsity = plot_data_heatmap(formatted_data, sparsity_plot = True, figsize = figsize)
    
    sums = plot_data_sums(formatted_data, average = average, log_scale = log_scale,
                               figsize = figsize)
    
    if formatted_data.dose is None:
        
        r_within = plot_data_within(formatted_data, average = average, 
                                                        log_scale = log_scale, 
                                                        suppress_warning = True,
                                                        figsize = figsize)
        
        return heatmap, sparsity, sums, r_within
        
    else:
        
        r_within, d_within, m_within = plot_data_within(formatted_data, average = average, 
                                                        log_scale = log_scale, 
                                                        suppress_warning = True,
                                                        figsize = figsize)
        
        return heatmap, sparsity, sums, r_within, d_within, m_within
    
        
        