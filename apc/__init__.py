# Main functions
from .format_data import format_data
from .get_design_components import get_design_components
from .get_design import get_design
from .get_data_sums import get_data_sums
from .fit_model import fit_model
from .fit_table import fit_table

# Plotting
from .plot_data_sums import plot_data_sums
from .plot_data_within import plot_data_within
from .plot_data_heatmap import plot_data_heatmap
from .plot_data_sparsity import plot_data_sparsity
from .plot_data_all import plot_data_all

#from matplotlib.pyplot import show

# Internal functions
from ._ReturnValue import _ReturnValue

# Implemented data sets
from .data_loss_TA import data_loss_TA
from .data_loss_VNJ import data_loss_VNJ
from .data_aids import data_aids
from .data_Belgian_lung_cancer import data_Belgian_lung_cancer
from .data_Italian_bladder_cancer import data_Italian_bladder_cancer
from .data_US_prostate_cancer import data_US_prostate_cancer
from .data_asbestos import data_asbestos