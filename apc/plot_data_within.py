import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_data_within(formatted_data, data_type = 'all', plot_type = 'all', 
                     group_size = None, average = False, log_scale = False, 
                     suppress_warning = False, figsize = None, title = None):
    
    """
    
    Plot for data sums or averages by age, period and cohort.
    
    Produces plots showing age, period and cohort sums. As a default this is done for 
    responses, doses and rates.
    
    
    Parameters
    ----------
    
    formatted_data : output of 'apc.format_data'
    
    data_type : {'response', 'dose', 'rate', 'mortality', 'all'}, optional
                Determines what figures are produced. 'mortlity' maps to 'rate'. If
                'all' is chosen and 'dose' is 'None', output is for 'response'.
    
    plot_type : {'all', 'awc', 'pwa', 'cwa', 'awp', 'pwc', 'cwp'}, optional
                Determines what subplots are produced. For example, 'awc' stands for 
                age within cohort. The first letter indicates what is on the x-axis, the
                last letter what groups are plotted.
    
    group_size : int, optional
                 age/periods/cohorts are grouped in into groups of size 'group_size'.
                 A warning is produced if this does not yield equal sized groups so the
                 last group is smaller than other groups. If 'None' this is set
                 internally, to the integer value of the ceiling of the number of time 
                 indices divided by 10.
    
    average : bool, optional
              Determines whether sums or averages are computed within the groups.
              (Default is 'False')
    
    
    log_scale : bool, optional
                Specifies whether the y-axis uses a log-scale.  (Default is 'False')
                
    title : str, optional
            Sets the canvas title. If 'None' this is set to "Sums of data by
            age/period/cohort" or "Averages of data by age/period/cohort" as
            appropriate.
            
    figsize : float tuple or list, optional
              Specifies the figure size. If left empty matplotlib determines this
              internally.
    
    suppress_warning : bool, optional.
                       Suppresses the warning for unbalanced group sizes resulting from
                       the aggregation.
                       
    
    Returns
    -------
    
    Matplotlib figure
    
    Examples
    --------
    
    >>> import apc
    >>> data = apc.data_Italian_bladder_cancer()
    >>> apc.plot_data_within(data)
    
    
    """
    
    if plot_type.lower() not in ['all', 'awp', 'awc', 'pwa', 'pwc', 'cwp', 'cwa']:
        
        raise ValueError('\'plot_type\' not recognised')
    
    if data_type.lower() not in ['response', 'dose', 'rate', 'mortality',
                                 'all']:
        
        raise ValueError('\'data_type\' not recognized.')
        
    if data_type.lower() in ['dose', 'rate', 'mortality'] and formatted_data.dose is None:
        
        raise ValueError('{} not available.'.format(data_type.title()))
    
    if data_type.lower() == 'all' and formatted_data.dose is None:
        data_type = 'response'
    
    data_as_vector = formatted_data.data_as_vector.reset_index()
    
    # Create figures. If data_type is 'all' we need three figures,
    # otherwise one.
    # If plot_type is 'all' we need 2x3 subplots, otherwise we need one.
    
    n_figs = 3 if data_type.lower() == 'all' else 1
    row_sub, col_sub = (2, 3) if plot_type.lower() == 'all' else (1, 1)
    
    figures = [plt.subplots(row_sub, col_sub, sharex = 'col', figsize = figsize,
                           ) for i in range(n_figs)]
    
    # Genereating lists to loop over
    if data_type.lower() == 'all':
        data_types =  ['Response', 'Dose', 'Rate']
    elif data_type.lower() in ['rate', 'mortality']:
        data_types = ['Rate']
    else:
        data_types = [data_type.title()]
        
    if plot_type.lower() == 'all':
        # The order here is important for a nice plot appearance
        plot_types = ['awc', 'pwa', 'cwa',
                      'awp', 'pwc', 'cwp']
    else:
        plot_types = [plot_type.lower()]
    
    abb = {'a' : 'Age', 'p' : 'Period', 'c' : 'Cohort'}
    
    agg_warning = False
    
    for n_fig, d_type in enumerate(data_types):
        
        # Pick the relevant axis and reshape it so we can easily iterate.
        if len(plot_types) == 1:
            tmp_axis = [figures[n_fig][1]]
            tmp_axis[0].set_ylabel(d_type)
        else:
            tmp_axis = figures[n_fig][1].reshape(1,len(plot_types))[0]
            tmp_axis[0].set_ylabel(d_type)
            tmp_axis[3].set_ylabel(d_type)
            
        # Set the canvas title
        if title is None:
            figures[n_fig][0].canvas.set_window_title(
                'Plot of {}s using two indices.'.format(d_type))
        else:
            figures[n_fig][0].canvas.set_window_title(title)

        
        for n_sub, plt_type in enumerate(plot_types):
            
            tmp_array = data_as_vector.pivot(index = abb[plt_type[0]], 
                                             columns = abb[plt_type[2]],
                                             values = d_type)
    
            # Do the aggregation
            
            tmp_n_col = tmp_array.shape[1]
    
            if group_size is None:
                tmp_agg = int(np.ceil(tmp_n_col/10))
            else:
                tmp_agg = group_size
            
            # This is quick and dirty to set the 0.0 only coming from sums to NaN
            #agg_array = pd.DataFrame([tmp_array.iloc[:,i:i + tmp_agg].sum(axis = 1) 
            #             if not average else 
            #             tmp_array.iloc[:,i:i + tmp_agg].mean(axis = 1) 
            #             for i in np.arange(0,tmp_n_col, tmp_agg)]).T
            agg_array_sum = pd.DataFrame(
                [tmp_array.iloc[:,i:i + tmp_agg].sum(axis = 1) 
                 for i in np.arange(0,tmp_n_col, tmp_agg)]).T
            agg_array_avg = pd.DataFrame(
                [tmp_array.iloc[:,i:i + tmp_agg].mean(axis = 1) 
                 for i in np.arange(0,tmp_n_col, tmp_agg)]).T
            
            agg_array_sum = agg_array_sum.where(agg_array_avg)
            
            agg_array = agg_array_avg if average else agg_array_sum 
            
            # Check for balance 
            # start contains the first, end the last column 
            # included in each sum / avg. We use this for labels later
            start = np.arange(0,tmp_n_col, tmp_agg)
            end = start + tmp_agg - 1 
            # Further, if the last included column is actually not contained
            # in the dataframe, we adjust the label and issue a warning later.
            if end[-1] > tmp_n_col - 1:
                end[-1] = tmp_n_col - 1
                agg_warning = True
            
            # Now create the column labels.
            if tmp_agg != 1:
                
                col_labels = ['{}-{}'.format(tmp_array.columns[start[i]], 
                                             tmp_array.columns[end[i]])
                             for i in range(len(start))]
                if start[-1] == end[-1]:
                    col_labels[-1] = tmp_array.columns[start[-1]]
                
            else:
                col_labels = tmp_array.columns
            
            agg_array.columns = col_labels
            agg_array.columns.name = abb[plt_type[2]]         
            
            # Now we can plot this, preferably on the right figure and subplot.
            tmp_title = 'within {}'.format(abb[plt_type[2]].lower())
            agg_array.plot(ax = tmp_axis[n_sub], logy = log_scale, title = tmp_title)
    
    
    for (fig, axis) in figures:
        fig.tight_layout()
        
    if agg_warning and not suppress_warning:
        print("Uneven aggregation: last group may be smaller than other groups")
    
    return figures[0][0] if len(figures) == 1 else [fig for (fig,axes) in figures]