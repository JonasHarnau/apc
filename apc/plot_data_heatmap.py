import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_data_heatmap(formatted_data, data_type = 'all', sparsity_plot = False, 
                      annot = False, space = None, n_x_ticks = None, 
                      n_y_ticks = None, title = None,  figsize = None):
    
    """
    
    Heatmap plot of data.
    
    Produces heatmap(s) of the data. As a default this is done for responses, doses and
    rates. The function has an argument ('sparsity_plot') that turns this into a sprsity
    plot. Furthermore, it allows to plot the heatmap with any two of age, period and
    cohort on the axes.
    
    
    Parameters
    ----------
    
    formatted_data : output of 'apc.format_data'
    
    data_type : {'response', 'dose', 'rate', 'mortality', 'all'}, optional
                Determines what the heatmaps are plotted for. 'mortlity' maps to 'rate'.
                If 'all' is chosen and 'dose' is 'None', output is for 'response'.
               
    sparsity_plot : bool or numeric, optional
                    Determines if a sparsity plot is made. If 'True' the sparsity cut
                    off is set to 1% of the average over the full data array. If this is
                    less than 1 it is set to 1. If a numeric input is made the sparsity
                    cut off is set to that value. (Default is 'False')
        
    annot : bool, optional
            Determines of the data values are plotted in the heatmap.
    
    space : {'AC', 'AP', 'PA', 'PC', 'CA', 'CP'}, optional
            Specifies what goes on the axes (A = Age, P = period, C = cohort). If 'None'
            this is set to the original data space.
        
    n_x_ticks : int, optional
                The number of ticks on the x-axis. If 'None' this is set to 10 if only
                one 'data_type' is plotted and to 5 otherwise.

    n_y_ticks : int, optional
                The number of ticks on the y-axis. If 'None' this is set to 10.    
    
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
    
    
    Notes
    -----
    
    Examples
    --------
    
    >>> import apc
    >>> data = apc.data_Italian_bladder_cancer()
    >>> apc.plot_data_heatmap(data)
    
    >>> import apc
    >>> data = apc.data_asbestos()
    >>> apc.plot_data_heatmap(data)
    >>> apc.plot_data_heatmap(data, sparsity_plot = True)
    
    """
    
    if data_type.lower() not in ['response', 'dose', 'rate', 'mortality',
                                 'all']:
        
        raise ValueError('\'data_type\' not recognized.')
        
    if data_type.lower() in ['dose', 'rate', 'mortality'] and formatted_data.dose is None:
        
        raise ValueError('{} not available.'.format(data_type.title()))
    
    if data_type.lower() == 'all' and formatted_data.dose is None:
        data_type = 'Response'
    
    if space is not None and space not in ['AC', 'AP', 'PA', 'PC', 'CA', 'CP']:
            raise ValueError('\'space\' not recognized.')
    
    if not any(isinstance(sparsity_plot,type) for type in [bool, int, float]):
        
        raise ValueError('\'sparsity_plot\' has to be any of bool, int and float')
    
    if data_type.lower() == 'all':
        data_types =  ['Response', 'Dose', 'Rate']
    elif data_type.lower() in ['rate', 'mortality']:
        data_types = ['Rate']
    else:
        data_types = [data_type.title()]
        
    data_as_vector = formatted_data.data_as_vector      
    
    if space is None:
        space = 'AC' if formatted_data.data_format in ['CL', 'trapezoid'] else formatted_data.data_format
        
    abb = {'A' : 'Age', 'P' : 'Period', 'C' : 'Cohort'}
    # Create a list of the relevant dataframes to loop over
    df_dict = {d_type: data_as_vector.reset_index().pivot(
        index = abb[space[0]], columns = abb[space[1]], values = d_type) for d_type in data_types}
     
    n_row, n_col = df_dict[data_types[0]].shape   
     
    if n_x_ticks is None:
        n_x_ticks = 10 if len(data_types) == 1 else 5
        
    if n_x_ticks in ['all', 'All'] or 2 *  n_x_ticks >= n_col:
        xticks = True
    else:
        xticks = int(n_col/n_x_ticks)
    
    if n_y_ticks is None:
        n_y_ticks = 10

    if n_y_ticks in ['all', 'All'] or 2 * n_y_ticks >= n_row:
        yticks = True
    else:
        yticks = int(n_row/n_y_ticks)
    
    fig, axes = plt.subplots(nrows = 1, ncols = 3 if data_type.lower() == 'all' else 1,
                            figsize = figsize)
    
    # For the loop below to work axes has to be a list
    if data_type.lower() != 'all':
        axes = [axes]
    
    for n_sub, d_type in enumerate(data_types):
        
        tmp_data = df_dict[d_type]
                
        if sparsity_plot is False:
            mask = None
            vmin = None
            vmax = None
        else:
            if sparsity_plot is True:
                # use 1% of total average if nothing is specified, 
                # unless that is less than 1 then use 1.
                sparse_cut_off = np.nanmean(tmp_data) * 0.01
                if sparse_cut_off < 1:
                    sparse_cut_off = 1
            # if a numeric value is supplied use that
            else:
                sparse_cut_off = sparsity_plot
                            
            vmin = 0
            vmax = sparse_cut_off
            mask = tmp_data[tmp_data.notnull()] > sparse_cut_off
            
        sns.heatmap(ax = axes[n_sub], data = tmp_data, annot = annot,
                   xticklabels = xticks, yticklabels = yticks, 
                   mask = mask, vmin = vmin, vmax = vmax)
        axes[n_sub].set_title(d_type)
    
    fig.tight_layout()
    
    if title is None:
        if sparsity_plot is False:
            title = 'Heatmap{}'.format('s' if len(data_types) > 1 else '')
        else:
            title = 'Sparsity plot{}'.format('s' if len(data_types) > 1 else '')
    
    fig.canvas.set_window_title(title)
    
    return fig