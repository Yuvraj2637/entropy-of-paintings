"""
generate_visualizations.py — Hydrodynamic Entropy of Paintings
(Verma & Chatterjee, Phys. Rev. Fluids 7, 114608, 2022 — adaptation)

NOTE ON ADAPTATION: The hydrodynamic entropy formula (H = -Σ p_k log2 p_k,
p_k = E(k)/ΣE(k)) is adapted from fluid kinetic-energy spectra (Euler turbulence)
to image Fourier spectral-power distributions. Spatial/pixel entropy is kept as a
secondary feature and is clearly labelled as Pixel/Intensity Entropy.

Generates:
1. 2D Aesthetic Space Map: Spectral Slope vs Normalized Hydrodynamic Entropy
2. Multi-Panel Dashboard: Slope, Hydrodynamic Entropy, Pixel Entropy distributions
3. ANOVA across all key metrics
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "spectral_entropy_results.csv"

# Color palette for art categories
CATEGORY_PALETTE = {
    "classical": "#1f77b4",  # Royal Blue (Order, Realism, Chiaroscuro)
    "modern": "#2ca02c",     # Emerald Green (Abstraction, Boldness)
    "picasso": "#d62728",    # Crimson Red (Cubist Structure, Contrast)
    "van_gogh": "#ff7f0e",   # Sunburst Gold/Orange (Vibrant Impasto)
    "other": "#9467bd",      # Deep Purple (Other Traditions: Ukiyo-e, Ravi Varma, Romanticism)
}

CATEGORY_NAMES = {
    "classical": "Classical Realism",
    "modern": "Modern / Abstract",
    "picasso": "Pablo Picasso",
    "van_gogh": "Vincent van Gogh",
    "other": "Other Traditions"
}


def confidence_ellipse(x, y, ax, n_std=1.96, facecolor="none", **kwargs):
    """
    Plots an empirical covariance confidence ellipse (default 95% CI ~ 1.96 std).
    """
    if len(x) != len(y) or len(x) == 0:
        return None

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_2d_aesthetic_space(df: pd.DataFrame, output_path: Path):
    """
    Creates a standalone, publication-quality 2D Aesthetic Space Map.
    X-axis: Spectral Slope (Scale-invariance / texture complexity)
    Y-axis: Normalized Hydrodynamic Entropy H_norm = H / H_max  [0, 1]
            (Adapted from Verma & Chatterjee 2022 — image Fourier power, not fluid KE)
    """
    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)

    # Plot individual artworks
    categories = [c for c in ["classical", "modern", "picasso", "van_gogh", "other"] if c in df["category"].unique()]
    for cat in categories:
        sub = df[df["category"] == cat]
        color = CATEGORY_PALETTE[cat]
        label = CATEGORY_NAMES[cat]

        # Scatter points
        ax.scatter(
            sub["spectral_slope"],
            sub["hydrodynamic_entropy_norm"],
            color=color,
            alpha=0.55,
            edgecolors="none",
            s=45,
            label=f"{label} (N={len(sub)})"
        )

        # 95% Confidence Ellipse
        confidence_ellipse(
            sub["spectral_slope"].values,
            sub["hydrodynamic_entropy_norm"].values,
            ax,
            n_std=1.96,
            edgecolor=color,
            linestyle="--",
            linewidth=1.8,
            alpha=0.85
        )

        # Group Centroid
        cx, cy = sub["spectral_slope"].mean(), sub["hydrodynamic_entropy_norm"].mean()
        ax.scatter(cx, cy, color=color, s=200, marker="P", edgecolors="black", linewidth=1.5, zorder=5)

        # Offsets for clean leader labels
        offsets = {
            "classical": (-130, 25),
            "modern": (20, 35),
            "picasso": (-135, -30),
            "van_gogh": (30, -25),
            "other": (25, -40)
        }
        ox, oy = offsets.get(cat, (20, 20))

        ax.annotate(
            f"{label}\n(m = {cx:.2f}, H_norm = {cy:.3f})",
            xy=(cx, cy),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#bbbbbb", linewidth=0.8),
            arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.0, shrinkA=3, shrinkB=6),
            zorder=6
        )

    # Reference line for 1/f^2 natural scene statistics
    ax.axvline(-2.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label=r"Natural $1/f^2$ Slope ($m=-2.0$)")

    ax.set_title(
        r"2D Aesthetic Space: Spectral Slope vs. Normalized Hydrodynamic Entropy ($N=400$ Artworks)"
        "\n[Adaptation: Verma & Chatterjee, PRFluids 2022 — image spectral power]",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel(r"Spectral Slope $m$  ($E(f) \sim f^m$, Log-Log Scale)", fontsize=12, labelpad=10)
    ax.set_ylabel(r"Normalized Hydrodynamic Entropy $H_{\rm norm} = H / H_{\rm max}$  [0, 1]", fontsize=12, labelpad=10)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10, loc="lower left")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[OK] 2D Aesthetic Space Map saved: {output_path}")


def plot_multipanel_dashboard(df: pd.DataFrame, output_path: Path):
    """
    Creates a 4-panel comprehensive comparative dashboard:
    Panel A: 2D Aesthetic Map with Centroids & Ellipses
    Panel B: Violin/Box Plot of Spectral Slope
    Panel C: Violin/Box Plot of Spatial Shannon Entropy
    Panel D: Violin/Box Plot of Normalized Spectral Entropy
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    categories = [c for c in ["classical", "modern", "picasso", "van_gogh", "other"] if c in df["category"].unique()]
    cat_labels = [CATEGORY_NAMES[c] for c in categories]
    colors = [CATEGORY_PALETTE[c] for c in categories]

    # -------------------------------------------------------------
    # Panel A: 2D Aesthetic Landscape (Slope vs Hydrodynamic Entropy)
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    for cat in categories:
        sub = df[df["category"] == cat]
        color = CATEGORY_PALETTE[cat]
        ax_a.scatter(
            sub["spectral_slope"],
            sub["hydrodynamic_entropy_norm"],
            color=color,
            alpha=0.5,
            edgecolors="none",
            s=35,
            label=CATEGORY_NAMES[cat]
        )
        confidence_ellipse(
            sub["spectral_slope"].values,
            sub["hydrodynamic_entropy_norm"].values,
            ax_a,
            n_std=1.96,
            edgecolor=color,
            linestyle="--",
            linewidth=1.6
        )
        cx, cy = sub["spectral_slope"].mean(), sub["hydrodynamic_entropy_norm"].mean()
        ax_a.scatter(cx, cy, color=color, s=120, marker="P", edgecolors="black", linewidth=1.2, zorder=5)

    ax_a.axvline(-2.0, color="dimgray", linestyle=":", linewidth=1.2, alpha=0.7, label="Natural Scale Invariance ($m=-2.0$)")
    ax_a.set_title(f"(A) Spectral Slope vs. Hydrodynamic Entropy ($N={len(df)}$)", fontsize=12, fontweight="bold")
    ax_a.set_xlabel(r"Spectral Slope $m$ (Log-Log)", fontsize=10)
    ax_a.set_ylabel(r"Hydrodynamic Entropy $H_{\rm norm}$ [0, 1]", fontsize=10)
    ax_a.grid(True, linestyle="--", alpha=0.35)
    ax_a.legend(fontsize=9, loc="lower left", framealpha=0.85)

    # Helper for styled violin + box plots
    def draw_violin_boxplot(ax, metric_col, title, ylabel, ref_line=None):
        data = [df[df["category"] == c][metric_col].values for c in categories]
        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor("black")
            pc.set_alpha(0.65)

        # Overlay compact boxplot
        bp = ax.boxplot(data, patch_artist=True, widths=0.18, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor("black")
            patch.set_linewidth(1.2)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.8)

        # Plot group means as red diamonds
        x_indices = list(range(1, len(categories) + 1))
        means = [np.mean(d) for d in data]
        ax.plot(x_indices, means, "rD", markersize=6, label="Group Mean", zorder=5)

        if ref_line is not None:
            ax.axhline(ref_line, color="dimgray", linestyle=":", linewidth=1.3, label=f"Ref ({ref_line})")

        ax.set_xticks(x_indices)
        ax.set_xticklabels(cat_labels, fontsize=9.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8.5, loc="best", framealpha=0.85)

    # -------------------------------------------------------------
    # Panel B: Spectral Slope
    # -------------------------------------------------------------
    draw_violin_boxplot(
        axes[0, 1],
        "spectral_slope",
        r"(B) Spectral Slope Distribution ($1/f^\alpha$ Power-Law)",
        "Spectral Slope m",
        ref_line=-2.0
    )

    # -------------------------------------------------------------
    # Panel C: Normalized Hydrodynamic Entropy (PRIMARY METRIC)
    # Adapted from Verma & Chatterjee 2022 — image Fourier power
    # -------------------------------------------------------------
    draw_violin_boxplot(
        axes[1, 0],
        "hydrodynamic_entropy_norm",
        r"(C) Normalized Hydrodynamic Entropy $H_{\rm norm}$ [PRIMARY]",
        r"$H_{\rm norm} = H / H_{\rm max}$  [0, 1]"
    )

    # -------------------------------------------------------------
    # Panel D: Pixel/Intensity Entropy (SECONDARY — NOT hydrodynamic)
    # Shannon entropy of pixel intensity histogram
    # -------------------------------------------------------------
    draw_violin_boxplot(
        axes[1, 1],
        "spatial_entropy",
        "(D) Pixel / Intensity Entropy [SECONDARY — not hydrodynamic]",
        "Pixel Entropy (Bits, 0–8)"
    )

    plt.suptitle(
        f"Hydrodynamic Entropy of Paintings — Multi-Panel Style Comparison ($N={len(df)}$)\n"
        r"[Verma \& Chatterjee PRFluids 2022 — adapted to image spectral power]",
        fontsize=14, fontweight="bold", y=0.998
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Multi-panel Dashboard saved: {output_path}")


def compute_statistical_summary(df: pd.DataFrame):
    """
    Computes ANOVA F-statistics and p-values across all categories.
    Primary metric: hydrodynamic_entropy_norm
    Secondary: spectral_slope, spatial_entropy (pixel/intensity)
    """
    print("\n" + "=" * 65)
    print("STATISTICAL INFERENCE & HYPOTHESIS TESTING (ONE-WAY ANOVA)")
    print("=" * 65)
    print("NOTE: hydrodynamic_entropy_norm is the PRIMARY metric (Verma & Chatterjee 2022)")
    print("      spatial_entropy = Pixel/Intensity Entropy (SECONDARY — not hydrodynamic)")
    print("=" * 65)

    metrics = [
        ("hydrodynamic_entropy_norm", "Hydro Entropy Norm [PRIMARY]"),
        ("shell_hydrodynamic_entropy_norm", "Shell Hydro Entropy Norm"),
        ("spectral_slope",               "Spectral Slope"),
        ("spatial_entropy",              "Pixel/Intensity Entropy [SECONDARY]"),
    ]
    categories = [c for c in ["classical", "modern", "picasso", "van_gogh", "other"] if c in df["category"].unique()]

    for col, label in metrics:
        if col not in df.columns:
            print(f"Metric: {label} — column '{col}' not found, skipping.")
            continue
        groups = [df[df["category"] == c][col].values for c in categories]
        f_val, p_val = stats.f_oneway(*groups)
        kw_val, kw_p = stats.kruskal(*groups)
        print(f"Metric: {label}")
        print(f"  One-way ANOVA   : F = {f_val:8.3f},  p = {p_val:.3e} {'(Statistically Significant)' if p_val < 0.05 else ''}")
        print(f"  Kruskal-Wallis  : H = {kw_val:8.3f},  p = {kw_p:.3e}")
        print("-" * 65)


def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found. Run analysis first.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} records from {CSV_PATH}.")

    # Output paths
    map_path = RESULTS_DIR / "2d_aesthetic_space_map.png"
    dashboard_path = RESULTS_DIR / "aesthetic_analysis_dashboard.png"

    # Generate Figures
    plot_2d_aesthetic_space(df, map_path)
    plot_multipanel_dashboard(df, dashboard_path)

    # Statistical Significance Testing
    compute_statistical_summary(df)


if __name__ == "__main__":
    main()
