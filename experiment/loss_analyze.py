# generate_appendix_figure_final.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

# --- 1. 加载数据 ---
try:
    df = pd.read_csv('sensitivity_analysis_results_with_zrf.csv')
except FileNotFoundError:
    raise RuntimeError("sensitivity_analysis_results_with_zrf.csv not found")

# --- 2. 风格与全局参数（单栏论文友好）---
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

color_zrf = '#1f77b4'
color_rrecall = '#ff7f0e'
color_rndcg = '#2ca02c'

# --- 3. 核心绘图函数 ---
def create_dual_axis_subplot(ax, data, param_name, xlabel_symbol,
                             show_y_left=True, show_y_right=True):
    if data.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return ([], []), ([], [])

    x = data[param_name]

    # 左轴：ZRF
    ax1 = ax
    ax1.plot(
        x, data['zrf_score'],
        marker='o', linewidth=2.0, markersize=5,
        color=color_zrf, label='ZRF Score'
    )
    if show_y_left:
        ax1.set_ylabel('ZRF Score (↑)', fontsize=14)
    ax1.set_xlabel(xlabel_symbol, fontsize=14)
    ax1.set_ylim(0.9, 1.0)
    ax1.tick_params(axis='both', labelsize=12)

    # 右轴：Retention
    ax2 = ax1.twinx()
    ax2.plot(
        x, data['retain_recall@20'],
        marker='s', linewidth=2.0, markersize=5,
        color=color_rrecall, label='Recall@20 (Retain)'
    )
    ax2.plot(
        x, data['retain_ndcg@20'],
        marker='^', linewidth=2.0, markersize=5,
        color=color_rndcg, label='NDCG@20 (Retain)'
    )

    if show_y_right:
        ax2.set_ylabel('Retention Metrics (↑)', fontsize=14)
    ax2.set_ylim(0.20, 0.40)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.grid(False)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    return ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()

# --- 4. 画布尺寸（单栏，高清）---
fig, axes = plt.subplots(
    1, 3,
    figsize=(8.5, 2.6),   # 单栏宽度
    dpi=300,
    sharey=True
)

# --- 5. 三个子图 ---
# Alpha
df_alpha = (
    df[(df['beta_sfd'] == 0.5) & (df['gamma_gkr'] == 0.5)]
    .drop_duplicates(subset=['alpha_prd'])
    .sort_values('alpha_prd')
)
(l1, la1), (l2, la2) = create_dual_axis_subplot(
    axes[0], df_alpha, 'alpha_prd', 'α', show_y_right=False
)

# Beta
df_beta = (
    df[(df['alpha_prd'] == 0.5) & (df['gamma_gkr'] == 0.5)]
    .drop_duplicates(subset=['beta_sfd'])
    .sort_values('beta_sfd')
)
create_dual_axis_subplot(
    axes[1], df_beta, 'beta_sfd', 'β',
    show_y_left=False, show_y_right=False
)

# Gamma
df_gamma = (
    df[(df['alpha_prd'] == 0.5) & (df['beta_sfd'] == 0.5)]
    .drop_duplicates(subset=['gamma_gkr'])
    .sort_values('gamma_gkr')
)
if df_gamma.empty:
    df_gamma = df_alpha.rename(columns={'alpha_prd': 'gamma_gkr'})

create_dual_axis_subplot(
    axes[2], df_gamma, 'gamma_gkr', 'γ', show_y_left=False
)

# --- 6. 统一图例（单栏友好）---
fig.legend(
    l1 + l2, la1 + la2,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    fontsize=14,
    frameon=False
)

plt.tight_layout(rect=[0, 0.08, 1, 1])

# --- 7. 保存 ---
plt.savefig(
    'sensitivity_appendix_figure.pdf',
    format='pdf',
    bbox_inches='tight'
)
plt.close()

print("sensitivity_appendix_figure.pdf generated successfully")
