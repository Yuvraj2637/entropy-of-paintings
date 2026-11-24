import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.spectral_analysis import (
    load_image,
    compute_total_power_spectrum,
    compute_shell_energy_and_entropy
)
from export_samples import safe_savefig

sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Select 2 representative paintings per category (10 total)
# For 'Other': 1 Japanese Ukiyo-e (The Great Wave) + 1 Raja Ravi Varma (Damayanti & Swan)
SAMPLE_SELECTIONS = [
    {
        "category": "Classical",
        "color": "#1f77b4",  # Blue
        "paintings": [
            {
                "name": "Mona Lisa (da Vinci)",
                "path": "sample_paintings_with_plots/classical/01_Mona_Lisa_painting.jpg",
                "linestyle": "-",
                "marker": "o"
            },
            {
                "name": "The Last Supper (da Vinci)",
                "path": "sample_paintings_with_plots/classical/02_The_Last_Supper_painting.jpg",
                "linestyle": "--",
                "marker": "s"
            }
        ]
    },
    {
        "category": "Modern",
        "color": "#2ca02c",  # Green
        "paintings": [
            {
                "name": "The Blue Rider (Kandinsky)",
                "path": "sample_paintings_with_plots/modern/01_The_Blue_Rider_painting.jpg",
                "linestyle": "-",
                "marker": "^"
            },
            {
                "name": "Houses at Murnau (Kandinsky)",
                "path": "sample_paintings_with_plots/modern/02_Houses_at_Murnau_painting.jpg",
                "linestyle": "--",
                "marker": "v"
            }
        ]
    },
    {
        "category": "Picasso",
        "color": "#d62728",  # Red
        "paintings": [
            {
                "name": "The Picador (1889)",
                "path": "sample_paintings_with_plots/picasso/01_The_Picador_painting.jpg",
                "linestyle": "-",
                "marker": "D"
            },
            {
                "name": "At the Lapin Agile (1905)",
                "path": "sample_paintings_with_plots/picasso/04_At_the_Lapin_Agile_painting.jpg",
                "linestyle": "--",
                "marker": "d"
            }
        ]
    },
    {
        "category": "Van Gogh",
        "color": "#ff7f0e",  # Orange
        "paintings": [
            {
                "name": "Beach at Scheveningen (1882)",
                "path": "sample_paintings_with_plots/van_gogh/01_Beach_at_Scheveningen_painting.jpg",
                "linestyle": "-",
                "marker": "p"
            },
            {
                "name": "Avenue of Poplars (1884)",
                "path": "sample_paintings_with_plots/van_gogh/04_Avenue_of_Poplars_painting.jpg",
                "linestyle": "--",
                "marker": "P"
            }
        ]
    },
    {
        "category": "Other",
        "color": "#9467bd",  # Purple
        "paintings": [
            {
                "name": "The Great Wave (Hokusai - Ukiyo-e)",
                "path": "data/images/other/japanese_ukiyo_e/01_Hokusai_Great_Wave.jpg",
                "linestyle": "-",
                "marker": "*"
            },
            {
                "name": "Damayanti & Swan (Raja Ravi Varma)",
                "path": "data/images/other/raja_ravi_varma/Damayanti_and_Swan.jpg",
                "linestyle": "--",
                "marker": "X"
            }
        ]
    }
]

print("Computing spectra for all 10 selected paintings (including Raja Ravi Varma)...")

computed_data = []

for cat_info in SAMPLE_SELECTIONS:
    cat_name = cat_info["category"]
    cat_color = cat_info["color"]
    for p_info in cat_info["paintings"]:
        p_path = Path(p_info["path"])
        if not p_path.exists():
            print(f"Error: {p_path} not found!")
            continue

        img = load_image(str(p_path))
        P = compute_total_power_spectrum(img)
        k_vals, E_k, S_bits, S_norm = compute_shell_energy_and_entropy(P)

        # Scale energy by total energy to get normalized shell distribution p(k) on a single scale
        E_tot = np.sum(E_k)
        p_k = E_k / E_tot

        # Inertial range fit: k in [6, 80]
        k_max = k_vals[-1]
        fit_mask = (k_vals >= 6) & (k_vals <= min(80, k_max))
        slope, intercept = np.polyfit(np.log10(k_vals[fit_mask]), np.log10(p_k[fit_mask]), 1)

        computed_data.append({
            "category": cat_name,
            "name": p_info["name"],
            "color": cat_color,
            "linestyle": p_info["linestyle"],
            "marker": p_info["marker"],
            "k_vals": k_vals,
            "p_k": p_k,
            "slope": slope,
            "h_norm": S_norm,
            "intercept": intercept
        })
        print(f"  {cat_name:10s} | {p_info['name']:36s} | slope m = {slope:6.2f} | H_norm = {S_norm:.3f}")

# ======================================================================
# FIGURE 1: ALL 10 PAINTINGS ON A SINGLE SCALE (SHARED AXIS)
# ======================================================================
fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

for d in computed_data:
    label_str = f"{d['category']}: {d['name']} (m = {d['slope']:.2f})"
    ax.plot(
        d["k_vals"], d["p_k"],
        color=d["color"],
        linestyle=d["linestyle"],
        linewidth=1.9,
        alpha=0.88,
        label=label_str
    )

# Highlight Inertial Subrange
ax.axvspan(6, 80, color="#fff3cd", alpha=0.35, label="Inertial Fitting Range $[6, 80]$")

# Reference Benchmarks anchored at k = 20
k_ref = np.linspace(1, 150, 200)
ref_anchor_k = 20
p_at_20 = [np.interp(20, d["k_vals"], d["p_k"]) for d in computed_data]
ref_anchor_y = np.median(p_at_20)

kolmogorov_ref = ref_anchor_y * (k_ref / ref_anchor_k) ** (-5.0 / 3.0)
natural_ref = ref_anchor_y * (k_ref / ref_anchor_k) ** (-2.0)

ax.plot(k_ref, kolmogorov_ref, color="#111111", linestyle="--", linewidth=2.2, label=r"Kolmogorov Turbulence ($k^{-5/3} \approx k^{-1.67}$)")
ax.plot(k_ref, natural_ref, color="#555555", linestyle=":", linewidth=2.2, label=r"Natural Scale-Invariance ($k^{-2.00}$)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(left=1, right=180)
ax.set_ylim(bottom=1e-6, top=0.5)

ax.set_xlabel(r"Wavenumber / Spatial Frequency $k$", fontsize=13, fontweight="bold")
ax.set_ylabel(r"Normalized Shell Energy $p(k) = E(k) / \sum E(k)$", fontsize=13, fontweight="bold")
ax.set_title(
    "Comparative Hydrodynamic Spectra Across Art Categories (Single Shared Scale)\n"
    "2 Representative Paintings per Category (Classical, Modern, Picasso, Van Gogh, Other)",
    fontsize=14, fontweight="bold", pad=12
)

# Clean 2-column legend
ax.legend(fontsize=8.5, loc="lower left", framealpha=0.95, ncol=2)
ax.grid(True, which="both", ls="--", alpha=0.35)

plt.tight_layout()
out_single = RESULTS_DIR / "single_scale_category_comparison.png"
safe_savefig(fig, out_single, dpi=300)
plt.close(fig)
print(f"\n[OK] Saved single-scale comparison plot: {out_single}")

# ======================================================================
# FIGURE 2: 5-PANEL COMPARATIVE GRID (SAME SCALE FOR ALL)
# ======================================================================
fig2, axes = plt.subplots(1, 5, figsize=(20, 4.5), dpi=300, sharex=True, sharey=True)

categories = ["Classical", "Modern", "Picasso", "Van Gogh", "Other"]

for i, cat in enumerate(categories):
    ax_sub = axes[i]
    cat_items = [d for d in computed_data if d["category"] == cat]
    
    for d in cat_items:
        clean_lbl = d['name'].split('(')[0].strip()
        ax_sub.plot(
            d["k_vals"], d["p_k"],
            color=d["color"],
            linestyle=d["linestyle"],
            linewidth=2.0,
            label=f"{clean_lbl}\n($m = {d['slope']:.2f}, H = {d['h_norm']:.3f}$)"
        )
    
    # Benchmarks in each subplot
    ax_sub.plot(k_ref, kolmogorov_ref, color="#111111", linestyle="--", linewidth=1.2, alpha=0.6, label="Kolmogorov $k^{-5/3}$")
    ax_sub.plot(k_ref, natural_ref, color="#555555", linestyle=":", linewidth=1.2, alpha=0.6, label="Natural $k^{-2}$")
    ax_sub.axvspan(6, 80, color="#fff3cd", alpha=0.3)
    
    ax_sub.set_xscale("log")
    ax_sub.set_yscale("log")
    ax_sub.set_xlim(left=1, right=180)
    ax_sub.set_ylim(bottom=1e-6, top=0.5)
    ax_sub.set_title(f"{cat}", fontsize=13, fontweight="bold")
    ax_sub.set_xlabel(r"Spatial Frequency $k$", fontsize=10, fontweight="bold")
    if i == 0:
        ax_sub.set_ylabel(r"Normalized Energy $p(k)$", fontsize=11, fontweight="bold")
    ax_sub.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    ax_sub.grid(True, which="both", ls="--", alpha=0.3)

fig2.suptitle("Category-by-Category Hydrodynamic Spectra (Identical Scales)", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
out_grid = RESULTS_DIR / "category_spectra_grid_5panel.png"
safe_savefig(fig2, out_grid, dpi=300)
plt.close(fig2)
print(f"[OK] Saved 5-panel comparative grid: {out_grid}")
