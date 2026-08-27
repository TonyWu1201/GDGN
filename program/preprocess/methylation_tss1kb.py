import pandas as pd
import numpy as np

# ===== 1. 建立 ID 映射：CCLE_ID → NAME =====
anno = pd.read_csv(
    "data/raw/cell_line_omics/Cell_lines_annotations_20181226.txt",
    sep="\t"
)
ccle_to_name = dict(zip(anno["CCLE_ID"], anno["Name"]))

common_cell_lines = pd.read_csv("data/common_cell_lines.csv")

# ===== 2. 读取甲基化数据（行=位点，列=细胞系） =====
meth = pd.read_csv(
    "data/raw/cell_line_omics/CCLE_DNA_methylation_TSS1kb.txt",
    sep="\t",
    index_col=0,        # locus_id 做行索引
    low_memory=False,
)

# 分离元数据列和细胞系列
meta_cols = ["CpG_sites_hg19", "avg_coverage"]
meta = meth[meta_cols].copy()
meth_data = meth.drop(columns=meta_cols)  # shape: (21338, 843)

# 将字符串型数值（含空白填充的 NaN/NA）转为真正的 numeric
meth_data = meth_data.apply(lambda col: col.str.strip() if col.dtype == object else col)
meth_data = meth_data.apply(pd.to_numeric, errors="coerce")

print(f"位点簇数: {meth_data.shape[0]}, 细胞系数: {meth_data.shape[1]}")

# ===== 3. 细胞系列名映射 =====
# 甲基化列名是 CCLE_ID (A549_LUNG)，映射到 Name
rename_map = {}
for col in meth_data.columns:
    if col in ccle_to_name:
        if ccle_to_name[col] in common_cell_lines["Name"].values:
            rename_map[col] = ccle_to_name[col]
    else:
        print(f"  [WARN] 未找到映射: {col}")

meth_data = meth_data.rename(columns=rename_map)
# 只保留能映射到标准名的细胞系
meth_data = meth_data[list(rename_map.values())]
print(f"映射后细胞系数: {meth_data.shape[1]}")

# ===== 4. 转置为 (细胞系 × 位点簇) =====
meth_matrix = meth_data.T  # shape: (843, 21338)

# ===== 6. 缺失值处理 =====
# 每列（位点）缺失率 > 30% → 丢弃
col_missing = meth_matrix.isna().mean()
keep_sites = col_missing[col_missing <= 0.3].index
meth_matrix = meth_matrix[keep_sites]
print(f"缺失率过滤后位点数: {meth_matrix.shape[1]}")

# ===== 7. 保存未填补、未标准化矩阵，供严格训练折内预处理 =====
meth_matrix = meth_matrix.sort_index()
meth_matrix.to_csv("data/processed/cell_line_omics/methylation_raw.csv")

# Legacy artifact for old checkpoints. Ver2 never reads this artifact in formal runs.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
meth_imputed = meth_matrix.fillna(meth_matrix.median())
meth_scaled = pd.DataFrame(
    scaler.fit_transform(meth_imputed),
    index=meth_matrix.index,
    columns=meth_matrix.columns,
).sort_index()

meth_scaled.to_csv("data/processed/cell_line_omics/methylation.csv")

# # ===== 8. 保存 =====
# np.save("data/processed/cell_line_omics/methylation.npy", meth_scaled.values.astype(np.float32))
# with open("data/processed/cell_line_omics/methylation_sites.txt", "w") as f:
#     for site in meth_scaled.columns:
#         f.write(f"{site}\n")
# # 保存细胞系顺序（用于与其他组学对齐）
# with open("data/processed/cell_line_omics/methylation_cells.txt", "w") as f:
#     for cell in meth_scaled.index:
#         f.write(f"{cell}\n")
