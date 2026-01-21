import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager

def setup_academic_style():
    """
    Sets professional matplotlib style.
    """
    try:
        # Try to use Times New Roman if available
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    except:
        pass
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['lines.markeredgewidth'] = 2.0
    # Adjusted for compact baseline, though main block overrides this
    plt.rcParams['figure.figsize'] = (4, 2) 

def plot_tradeoff_scatter_enhanced(ax, data, title, utility_metric, x_limits=None, y_limits=None, is_NGCF=False, show_ylabel=False):
    """
    Draws an enhanced tradeoff scatter plot.
    """
    methods = ['Origin', 'Retrain', 'GNNdelete', 'UnlearnRec', 'DPU']
    colors = ['#A9A9A9', '#0072B2', '#E69F00', '#D55E00', '#009E73']  # Grey, Blue, Orange, Red, Green
    markers = ['o', 's', '^',  'D', '*']
    
    unlearning_metric = 'MIA ACC'
    points = {}

    # 1. Plot scatter points
    for i, method in enumerate(methods):
        if method not in data:
            continue
        x = data[method][utility_metric]
        y = data[method][unlearning_metric]
        points[method] = (x, y)
        
        ax.scatter(x, y, 
                   color=colors[i], 
                   marker=markers[i], 
                   s=250, 
                   label=method, 
                   alpha=0.9,
                   edgecolors='black', 
                   linewidth=1.5,
                   zorder=10)

    # Apply limits first so text placement calculations are accurate relative to the view
    if x_limits:
        ax.set_xlim(x_limits)
    if y_limits:
        ax.set_ylim(y_limits)

    # 2. Add text labels for points (Dynamic offset based on limits)
    xlim_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    ylim_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    
    x_offset_base = xlim_range * 0.02
    y_offset_base = ylim_range * 0.03

    offsets = {
        'Origin': (x_offset_base - 0.01, y_offset_base),
        'Retrain': (x_offset_base - 0.012, -y_offset_base - 0.05),
        'GNNdelete': (-x_offset_base+0.007, y_offset_base - 0.02), # Move left to avoid overlap
        'UnlearnRec': (x_offset_base+0.005, y_offset_base*0.5), # Move left to avoid overlap
        'DPU': (x_offset_base+0.005, y_offset_base)
    }
    if title.endswith('(NGCF)'):
        offsets['GNNdelete'] = (-x_offset_base+0.005, y_offset_base-0.08) # Special adjustment for NGCF
        offsets['Origin'] = (x_offset_base-0.004, y_offset_base) 
        offsets['Retrain'] = (x_offset_base + 0.002, -y_offset_base + 0.025) 
        offsets['DPU'] = (x_offset_base+0.002, y_offset_base + 0.02)
    for method, (x, y) in points.items():
        dx, dy = offsets.get(method, (x_offset_base, y_offset_base))
        
        # Specific adjustments to avoid overlapping with borders or other points
        if method == 'Origin':
             ha = 'right'
        else:
             ha = 'left'
             
        # Only show text for DPU, Retrain, Origin to reduce clutter if needed, or adjust positions better
        # Based on user request, showing all
        if method == 'GNNdelete' and is_NGCF: # Special adjustment for NGCF GNNdelete overlap
             dy += 0.04
        
        ax.text(x + dx, y + dy, method, 
                fontsize=14, 
                # fontweight='bold', 
                va='center',
                ha=ha,
                color=colors[methods.index(method)],
                zorder=12)

    # 3. Add Auxiliary Lines (Quadrants)
    
    ceu_full_x, ceu_full_y = points['Retrain']
    origin_x, origin_y = points['Origin']
    
    # Find max utility among ablation/baselines
    other_utilities = [points[m][0] for m in ['GNNdelete', 'UnlearnRec', 'DPU'] if m in points]
    max_utility_ablations = max(other_utilities) if other_utilities else ceu_full_x
    
    # Midpoint for vertical line
    mid_x = (ceu_full_x + max_utility_ablations) / 2
    
    if title.endswith('(NGCF)'):
        mid_x -= xlim_range * 0.02 + 0.004
    else:
        mid_x -= xlim_range * 0.02 + 0.01
    ax.axvline(x=mid_x, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=5)
    
    # Horizontal Line
    unlearned_mias = [points[m][1] for m in ['Retrain', 'GNNdelete','UnlearnRec', 'DPU'] if m in points]
    max_mia_unlearned = max(unlearned_mias) if unlearned_mias else origin_y
    
    mid_y = (origin_y + max_mia_unlearned) / 2 
    
    ax.axhline(y=mid_y, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=5)

    # 4. Quadrant Annotations
    curr_xlim = ax.get_xlim()
    curr_ylim = ax.get_ylim()
    
    x_center_left = curr_xlim[0] + (mid_x - curr_xlim[0]) / 2
    x_center_right = mid_x + (curr_xlim[1] - mid_x) / 2
    y_center_top = mid_y + (curr_ylim[1] - mid_y) / 2
    y_center_bottom = curr_ylim[0] + (mid_y - curr_ylim[0]) / 2
    
    text_kwargs = dict(ha='center', va='center', fontsize=17, alpha=1, fontweight='bold')

    
            


    if title.endswith('(NGCF)'):
        ax.text(x_center_right, y_center_top-0.06, 'Not Unlearned', 
            color='gray', **text_kwargs)

        ax.text(x_center_left, y_center_top-0.06, 'Worst Case', 
            color='gray', **text_kwargs)
        
        ax.text(x_center_left, y_center_bottom+0.14, 'Destructive', 
            color='#D55E00', **text_kwargs)

        ax.text(x_center_right, y_center_bottom+0.14, 'Ideal Region', 
            color='#009E73',
            **text_kwargs)
    else:
        ax.text(x_center_right, y_center_top-0.06, 'Not Unlearned', 
            color='gray', **text_kwargs)

        ax.text(x_center_left, y_center_top-0.06, 'Worst Case', 
            color='gray', **text_kwargs)
        
        ax.text(x_center_left, y_center_bottom+0.1, 'Destructive', 
            color='#D55E00', **text_kwargs)

        ax.text(x_center_right, y_center_bottom+0.1, 'Ideal Region', 
            color='#009E73',
            **text_kwargs)
    
    ax.set_title(title, weight='bold', fontsize=15, pad=15)
    ax.set_xlabel(f'Model Utility ({utility_metric.upper()}) $\\rightarrow$', fontsize=15)
    
    if show_ylabel:
        ax.set_ylabel(f'Privacy Risk (MIA Acc) $\\downarrow$', fontsize=15)

    ax.grid(True, linestyle=':', alpha=1)

# --- Prepare Data ---
plot_data = {
    'LightGCN': {
        'Origin': {'ndcg@10': 0.2682, 'recall@10': 0.1754, 'MIA ACC': 0.8967},
        'Retrain': {'ndcg@10': 0.2671, 'recall@10': 0.1332, 'MIA ACC': 0.4987},
        'GNNdelete': {'ndcg@10': 0.2161, 'recall@10': 0.0742, 'MIA ACC': 0.4893}, 
        'UnlearnRec': {'ndcg@10': 0.2161, 'recall@10': 0.0751, 'MIA ACC': 0.5531}, 
        'DPU': {'ndcg@10': 0.2412, 'recall@10': 0.1587, 'MIA ACC': 0.5314}       
    },
    'NGCF': { 
        'Origin': {'ndcg@10': 0.2682, 'recall@10': 0.1923 , 'MIA ACC': 0.9001},
        'Retrain': {'ndcg@10': 0.2671, 'recall@10': 0.1723 , 'MIA ACC': 0.4793},
        'GNNdelete': {'ndcg@10': 0.2161, 'recall@10': 0.1515, 'MIA ACC': 0.6872}, 
        'UnlearnRec': {'ndcg@10': 0.2161, 'recall@10': 0.1508 , 'MIA ACC': 0.5737}, 
        'DPU': {'ndcg@10': 0.2412, 'recall@10': 0.1958, 'MIA ACC': 0.4392}   
    }
}

K_VALUE = '10'
metric_hr = f'recall@{K_VALUE}'

# --- Plotting ---
setup_academic_style()

# MODIFIED: Reduced figsize and added constrained layout for compactness
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), layout='constrained')

# Plot (a) LightGCN
# Adjusted x_limits to fit data better (min is around 0.07)
plot_tradeoff_scatter_enhanced(ax1, plot_data['LightGCN'], 
                               f'(a) Recall@{K_VALUE} vs. MIA (LightGCN)', 
                               utility_metric=metric_hr, 
                               x_limits=(0.06, 0.19), 
                               y_limits=(0.40, 1.0),
                               is_NGCF=False, 
                               show_ylabel=True)

# Plot (b) NGCF
# Adjusted x_limits to fit data better (max is around 0.20)
plot_tradeoff_scatter_enhanced(ax2, plot_data['NGCF'], 
                               f'(b) Recall@{K_VALUE} vs. MIA (NGCF)', 
                               utility_metric=metric_hr, 
                               x_limits=(0.14, 0.21), 
                               y_limits=(0.40, 1.0),
                               is_NGCF=True, 
                               show_ylabel=False) 

# MODIFIED: Legend position adjusted for constrained layout
handles, labels = ax1.get_legend_handles_labels()
# Filter unique handles/labels
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.12), fontsize=15, frameon=False)

plt.savefig('mia.pdf', bbox_inches='tight')