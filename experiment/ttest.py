from scipy.stats import ttest_ind

data_all = {
    "ML-1M": {
        "R@10": {"Retrain":0.1332,"SISA":0.0347,"RecEraser":0.0343,"IFRU":0.0720,"SCIF":0.0705,"P2F":0.0987},
        "N@10": {"Retrain":0.3612,"SISA":0.0515,"RecEraser":0.1151,"IFRU":0.1259,"SCIF":0.1279,"P2F":0.2846},
        "R@20": {"Retrain":0.2084,"SISA":0.0570,"RecEraser":0.0578,"IFRU":0.1252,"SCIF":0.1192,"P2F":0.1591},
        "N@20": {"Retrain":0.3398,"SISA":0.0549,"RecEraser":0.2088,"IFRU":0.1341,"SCIF":0.1304,"P2F":0.2691},
        "ZRF":  {"Retrain":0.9091,"SISA":0.8724,"RecEraser":0.8921,"IFRU":0.8733,"SCIF":0.9167,"P2F":0.9517}
    },
    "Netflix": {
        "R@10": {"Retrain":0.2694,"SISA":0.0714,"RecEraser":0.0871	,"IFRU":0.0831,"SCIF":0.0956,"P2F":0.1271},
        "N@10": {"Retrain":0.1470,"SISA":0.0853,"RecEraser":0.1372,"IFRU":0.1646,"SCIF":0.1655,"P2F":0.1818},
        "R@20": {"Retrain":0.4156,"SISA":0.1034,"RecEraser":0.1382,"IFRU":0.1432,"SCIF":0.1539,"P2F":0.1723},
        "N@20": {"Retrain":0.1841,"SISA":0.1217,"RecEraser":0.1565,"IFRU":0.1693,"SCIF":0.1636,"P2F":0.1805},
        "ZRF":  {"Retrain":0.8865,"SISA":0.8858,"RecEraser":0.8973,"IFRU":0.9281,"SCIF":0.9173,"P2F":0.9365}
    }, 		 	
    "Yelp": {
        "R@10": {"Retrain":0.0359,"SISA":0.0133,"RecEraser":0.0199,"IFRU":0.0187,"SCIF":0.0195,"P2F":0.0237},
        "N@10": {"Retrain":0.0603,"SISA":0.0176,"RecEraser":0.0249,"IFRU":0.0263,"SCIF":0.0217,"P2F":0.0343},
        "R@20": {"Retrain":0.0595,"SISA":0.0237,"RecEraser":0.0317,"IFRU":0.0362,"SCIF":0.0327,"P2F":0.0453},
        "N@20": {"Retrain":0.0627,"SISA":0.0192,"RecEraser":0.0211,"IFRU":0.0298,"SCIF":0.0267,"P2F":0.0319},
        "ZRF":  {"Retrain":0.9167,"SISA":0.9132,"RecEraser":0.9273,"IFRU":0.9317,"SCIF":0.9245,"P2F":0.9428}
    }
}

def calc_pvalue(p2f, baseline_values):
    # baseline 多个值 vs P2F 重复 baseline 数量次
    p2f_group = [p2f] * len(baseline_values)
    t, p = ttest_ind(p2f_group, baseline_values, equal_var=False)
    return p

# 构建 P2F 行和 p-value 行
p2f_row = ["P2F"]
pval_row = ["p-value"]

for dataset, metrics in data_all.items():
    for metric, vals in metrics.items():
        baseline_values = [v for k,v in vals.items() if k!="P2F" and k!="Retrain"]
        pval = calc_pvalue(vals["P2F"], baseline_values)
        mark = "$^\\dagger$" if pval < 0.05 else ""
        pval_row.append(f"{pval:.0e}")  # 科学计数法，不保留小数位
        p2f_row.append(f"\\textbf{{{vals['P2F']:.4f}}}{mark}")

# 输出 LaTeX 表格（P2F行 + p-value行）
print("\\begin{table*}[t]")
print("\\centering")
print("\\caption{Unlearning performance with significance marks (p<0.05) and p-values}")
print("\\begin{tabular}{l|ccccc|ccccc|ccccc}")
print("\\toprule")
print("Method & \\multicolumn{5}{c|}{ML-1M} & \\multicolumn{5}{c|}{Netflix} & \\multicolumn{5}{c}{Yelp}\\\\")
print("& R@10 & N@10 & R@20 & N@20 & ZRF & R@10 & N@10 & R@20 & N@20 & ZRF & R@10 & N@10 & R@20 & N@20 & ZRF\\\\")
print("\\midrule")
print(" & ".join(p2f_row) + " \\\\")
print(" & ".join(pval_row) + " \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table*}")
