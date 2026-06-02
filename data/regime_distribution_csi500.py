import pandas as pd
import numpy as np
import json

df_train = pd.read_parquet("data/processed/csi500_train_with_regime.parquet")
df_val   = pd.read_parquet("data/processed/csi500_val_with_regime.parquet")
df_test  = pd.read_parquet("data/processed/csi500_test_with_regime.parquet")

train_dist = df_train.groupby("regime_id").size() / len(df_train)
val_dist   = df_val.groupby("regime_id").size()   / len(df_val)
test_dist  = df_test.groupby("regime_id").size()  / len(df_test)
shift      = val_dist - train_dist

print("Train regime分布:")
print(train_dist.round(3))
print("\nVal regime分布:")
print(val_dist.round(3))
print("\nTest regime分布:")
print(test_dist.round(3))
print("\n分布偏移 (val - train):")
print(shift.round(3))

# 存起来
result = {
    "train": train_dist.round(4).to_dict(),
    "val":   val_dist.round(4).to_dict(),
    "test":  test_dist.round(4).to_dict(),
    "shift_val_minus_train": shift.round(4).to_dict(),
}
with open("downstream/results/regime_distribution_csi500.json", "w") as f:
    json.dump(result, f, indent=2)
print("\n✅ 已保存到 downstream/results/regime_distribution_csi500.json")