import pandas as pd

df = pd.read_csv("results/spectral_entropy_results.csv")
for cat, g in df.groupby("category"):
    h  = g["hydrodynamic_entropy_norm"].mean()
    hs = g["hydrodynamic_entropy_norm"].std()
    sl = g["spectral_slope"].mean()
    pe = g["spatial_entropy"].mean()
    print(f"{cat:12s}: H_norm={h:.4f} ± {hs:.4f}   slope={sl:.4f}   pixel_entropy={pe:.4f}")
print(f"\nTotal: {len(df)} images")
