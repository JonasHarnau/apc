from .plot_data_heatmap import plot_data_heatmap

def plot_data_sparsity(formatted_data, sparsity_cut_off = None, *args, **kwargs):
    
    """
    
    Produces a sparsity plot of the data.
    
    This is a wrapper for 'plot_data_heatmap' except 'sparsity_cut_off' is set to 'True'
    by default. See the documentation for 'plot_data_heatmap' for more information.
    
    """
    
    if sparsity_cut_off is not None and not any(isinstance(sparsity_cut_off,type) for type in [bool, int, float]):
        
        raise ValueError('\'sparsity_cut_off\' has to be either numeric or \'None\'.')
   
    
    if sparsity_cut_off is None:
        sparsity_cut_off = True
        
    return plot_data_heatmap(formatted_data = formatted_data, sparsity_plot =
                             sparsity_cut_off, *args, **kwargs)
    
    