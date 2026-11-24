"""
export_samples.py

Creates a curated showcase folder 'sample_paintings_with_plots/' inside idc_project.
Contains 5 representative paintings per category (Classical, Modern, Picasso, Van Gogh):
  - Original painting image
  - Individual hydrodynamic spectral analysis plot (with Kolmogorov & 1/f^2 reference lines)
  - Side-by-side comparison figure (Painting on Left, Hydrodynamic Plot on Right with metrics)
  - samples_summary.csv with all computed hydrodynamic and aesthetic metrics
  - README.md with full descriptions for presentation and reporting
"""

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from src.spectral_analysis import (
    load_image,
    compute_total_power_spectrum,
    compute_shell_energy_and_entropy,
    compute_spatial_entropy,
)

# Root folders
PROJECT_ROOT = Path("c:/Users/ACER/OneDrive/Desktop/Placement26/Projects/idc_project")
OUTPUT_ROOT = PROJECT_ROOT / "sample_paintings_with_plots"
CSV_PATH = PROJECT_ROOT / "results" / "spectral_entropy_results.csv"

# 5 Curated Representative Paintings per category
CURATED_SAMPLES = {
    "classical": [
        {
            "filename": "01_Mona_Lisa.jpg",
            "display_name": "Leonardo da Vinci - Mona Lisa (c. 1503)",
            "short_id": "01_Mona_Lisa",
        },
        {
            "filename": "02_The_Last_Supper_-_Leonardo_Da_Vinci_-_High_Resolution_32x16.jpg",
            "display_name": "Leonardo da Vinci - The Last Supper (1498)",
            "short_id": "02_The_Last_Supper",
        },
        {
            "filename": "08_Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg",
            "display_name": "Sandro Botticelli - The Birth of Venus (c. 1485)",
            "short_id": "03_Birth_of_Venus",
        },
        {
            "filename": "95_Raphaels_Triumph_of_Galatea_02.jpg",
            "display_name": "Raphael - Triumph of Galatea (1512)",
            "short_id": "04_Triumph_of_Galatea",
        },
        {
            "filename": "98_Raphael_-_The_Sistine_Madonna_-_Google_Art_Project.jpg",
            "display_name": "Raphael - The Sistine Madonna (1512)",
            "short_id": "05_Sistine_Madonna",
        },
    ],
    "modern": [
        {
            "filename": "38_Kandinsky-Blue_Rider.jpg",
            "display_name": "Wassily Kandinsky - Der Blaue Reiter (1903)",
            "short_id": "01_The_Blue_Rider",
        },
        {
            "filename": "100_Vasily_Kandinsky_-_Houses_at_Murnau_-_199433_-_Art.jpg",
            "display_name": "Wassily Kandinsky - Houses at Murnau (1909)",
            "short_id": "02_Houses_at_Murnau",
        },
        {
            "filename": "44_Kandinsky_Sunday.jpg",
            "display_name": "Wassily Kandinsky - Sunday / Old Russia (1904)",
            "short_id": "03_Sunday_Old_Russia",
        },
        {
            "filename": "87_Wassily_Kandinsky_-_The_Waterfall_-_1941529_-_Yale.jpg",
            "display_name": "Wassily Kandinsky - The Waterfall (1909)",
            "short_id": "04_The_Waterfall",
        },
        {
            "filename": "12_Vassily_Kandinsky_1898_-_Odessa_Port.jpg",
            "display_name": "Wassily Kandinsky - Odessa Port (1898)",
            "short_id": "05_Odessa_Port",
        },
    ],
    "picasso": [
        {
            "filename": "01_The_picador.jpg",
            "display_name": "Pablo Picasso - The Picador (1889)",
            "short_id": "01_The_Picador",
        },
        {
            "filename": "07_The_barefoot_girl.jpg",
            "display_name": "Pablo Picasso - The Barefoot Girl (1895)",
            "short_id": "02_The_Barefoot_Girl",
        },
        {
            "filename": "11_Acrobat_and_Young_Harlequin.jpg",
            "display_name": "Pablo Picasso - Acrobat and Young Harlequin (1905)",
            "short_id": "03_Acrobat_and_Harlequin",
        },
        {
            "filename": "12_Au_Lapin_Agile.jpg",
            "display_name": "Pablo Picasso - At the Lapin Agile (1905)",
            "short_id": "04_At_the_Lapin_Agile",
        },
        {
            "filename": "14_Brick_Factory_at_Tortosa.jpg",
            "display_name": "Pablo Picasso - Brick Factory at Tortosa (1909)",
            "short_id": "05_Brick_Factory_Tortosa",
        },
    ],
    "van_gogh": [
        {
            "filename": "12_Vincent_van_Gogh_-_Beach_at_Scheveningen_in_calm_w.jpg",
            "display_name": "Vincent van Gogh - Beach at Scheveningen (1882)",
            "short_id": "01_Beach_at_Scheveningen",
        },
        {
            "filename": "28_Potato_Digging_Five_Figures.jpg",
            "display_name": "Vincent van Gogh - Potato Digging, Five Figures (1883)",
            "short_id": "02_Potato_Digging",
        },
        {
            "filename": "45_The_Old_Tower_at_Nuenen_-_Vincent_Van_Gogh.jpg",
            "display_name": "Vincent van Gogh - The Old Tower at Nuenen (1884)",
            "short_id": "03_Old_Tower_at_Nuenen",
        },
        {
            "filename": "70_Vincent_van_Gogh_-_Avenue_of_poplars_in_autumn_-_G.jpg",
            "display_name": "Vincent van Gogh - Avenue of Poplars in Autumn (1884)",
            "short_id": "04_Avenue_of_Poplars",
        },
        {
            "filename": "76_Vincent_van_Gogh_-_Water_mill_at_Opwetten_1884.jpg",
            "display_name": "Vincent van Gogh - Water Mill at Opwetten (1884)",
            "short_id": "05_Water_Mill_at_Opwetten",
        },
    ],
}


def safe_savefig(fig, output_path: Path, dpi: int = 300):
    import time
    for attempt in range(5):
        try:
            fig.savefig(output_path, dpi=dpi)
            return
        except OSError:
            if attempt < 4:
                time.sleep(0.5)
            else:
                raise


def render_spectral_plot(image_path: Path, output_plot_path: Path, display_title: str):
    """
    Renders high-resolution hydrodynamic energy spectrum plot with Kolmogorov and 1/f^2 reference lines
    following Verma & Chatterjee (Phys. Rev. Fluids, 2022):
      E(k) = sum_{|k'|=k} Power(k')
      p(k) = E(k) / sum_k E(k)
      S    = - sum_k p(k) * log2(p(k))
    """
    image = load_image(str(image_path))
    total_power = compute_total_power_spectrum(image)
    k_vals, E_k, S_bits, S_norm = compute_shell_energy_and_entropy(total_power)

    # Fit power-law on the physical inertial range [6, 80]
    # (excludes k < 6 canvas boundary/framing effects and k > 80 sensor/craquelure noise floor)
    k_max = k_vals[-1]
    fit_mask = (k_vals >= 6) & (k_vals <= min(80, k_max))

    log_k_fit = np.log10(k_vals[fit_mask])
    log_E_fit = np.log10(E_k[fit_mask])

    slope, intercept = np.polyfit(log_k_fit, log_E_fit, deg=1)
    fitted_line = 10 ** (intercept + slope * np.log10(k_vals))

    # Anchored benchmark lines at center of the inertial range (k = 25)
    k_mid = 25
    E_mid = 10 ** (intercept + slope * np.log10(k_mid))
    kolmogorov_line = E_mid * (k_vals / k_mid) ** (-5.0 / 3.0)
    natural_line = E_mid * (k_vals / k_mid) ** (-2.0)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(k_vals, E_k, color="#111111", linewidth=2.0, label=r"Total Energy Spectrum $E(k)$")
    ax.plot(k_vals, fitted_line, color="#ff7f00", linestyle="-", linewidth=2.2, label=f"Fitted Slope: $m = {slope:.2f}$")
    ax.plot(k_vals, kolmogorov_line, color="#984ea3", linestyle="--", linewidth=1.6, alpha=0.85,
            label=r"Kolmogorov Turbulence ($k^{-5/3} \approx k^{-1.67}$)")
    ax.plot(k_vals, natural_line, color="#555555", linestyle=":", linewidth=1.6, alpha=0.85,
            label=r"Natural Scale-Invariance ($k^{-2.00}$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber / Spatial Frequency $k$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Shell Energy $E(k)$", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Hydrodynamic Spectrum: {display_title}\n"
        f"Slope $m = {slope:.2f}$ | $H_{{\\rm norm}} = {S_norm:.3f}$ ({S_bits:.2f} bits)",
        fontsize=11, fontweight="bold", pad=10
    )
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.92)
    ax.grid(True, which="both", ls="--", alpha=0.35)

    plt.tight_layout()
    safe_savefig(fig, output_plot_path, dpi=300)
    plt.close(fig)

    return {
        "slope": float(slope),
        "h_norm": S_norm,
        "h_bits": S_bits,
    }


def create_side_by_side(painting_path: Path, plot_path: Path, output_combined_path: Path, title: str, metrics: dict):
    """
    Creates a composite figure: Painting on Left, Hydrodynamic Spectral Plot on Right.
    Ready for presentation slides.
    """
    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300, gridspec_kw={"width_ratios": [1, 1.25]})

    # Left: Painting
    img = Image.open(painting_path).convert("RGB")
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title("Artwork", fontsize=12, fontweight="bold", pad=8)

    # Right: Plot image
    plot_img = Image.open(plot_path).convert("RGB")
    ax_plot.imshow(plot_img)
    ax_plot.axis("off")
    ax_plot.set_title("Hydrodynamic Spectrum & Benchmarks", fontsize=12, fontweight="bold", pad=8)

    fig.suptitle(
        f"{title}\n"
        f"Normalized Hydrodynamic Entropy $H_{{\\rm norm}} = {metrics['h_norm']:.3f}$  |  "
        f"Spectral Slope $m = {metrics['slope']:.2f}$  |  "
        f"Pixel Entropy $H_{{\\rm pixel}} = {metrics.get('spatial_entropy', 0.0):.2f}$ bits",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    safe_savefig(fig, output_combined_path, dpi=300)
    plt.close(fig)


def main():
    print(f"Creating sample showcase in: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Read existing CSV for pre-calculated numbers if available
    df_all = None
    if CSV_PATH.exists():
        df_all = pd.read_csv(CSV_PATH)

    summary_records = []

    for category, samples in CURATED_SAMPLES.items():
        cat_dir = OUTPUT_ROOT / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing category: {category.upper()} (5 samples)")

        for s in samples:
            src_image_path = PROJECT_ROOT / "data" / "images" / category / s["filename"]
            if not src_image_path.exists():
                print(f"  [MISSING] Image not found: {src_image_path}")
                continue

            short_id = s["short_id"]
            dst_painting_path = cat_dir / f"{short_id}_painting.jpg"
            dst_plot_path = cat_dir / f"{short_id}_spectrum.png"
            dst_combined_path = cat_dir / f"{short_id}_side_by_side.png"

            # 1. Copy painting
            shutil.copy2(src_image_path, dst_painting_path)

            # 2. Render dedicated spectral plot
            metrics = render_spectral_plot(src_image_path, dst_plot_path, s["display_name"])

            # Add spatial entropy from CSV or calculate
            if df_all is not None:
                row = df_all[(df_all["category"] == category) & (df_all["image_name"] == s["filename"])]
                if not row.empty:
                    metrics["spatial_entropy"] = float(row["spatial_entropy"].values[0])
                    metrics["shell_hydrodynamic_entropy_norm"] = float(row["shell_hydrodynamic_entropy_norm"].values[0])
            if "spatial_entropy" not in metrics:
                img_arr = load_image(str(src_image_path))
                metrics["spatial_entropy"] = float(np.mean([compute_spatial_entropy(img_arr[:, :, c]) for c in range(3)]))

            # 3. Create Side-by-Side comparison
            create_side_by_side(dst_painting_path, dst_plot_path, dst_combined_path, s["display_name"], metrics)

            print(f"  [OK] {s['display_name']} -> m={metrics['slope']:.2f}, H_norm={metrics['h_norm']:.3f}")

            summary_records.append({
                "category": category,
                "short_id": short_id,
                "artwork_title": s["display_name"],
                "source_file": s["filename"],
                "painting_file": f"{category}/{dst_painting_path.name}",
                "spectrum_plot_file": f"{category}/{dst_plot_path.name}",
                "side_by_side_file": f"{category}/{dst_combined_path.name}",
                "hydrodynamic_entropy_norm": round(metrics["h_norm"], 4),
                "hydrodynamic_entropy_bits": round(metrics["h_bits"], 3),
                "spectral_slope_m": round(metrics["slope"], 3),
                "pixel_spatial_entropy": round(metrics.get("spatial_entropy", 0.0), 3),
            })

    # Save summary CSV
    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = OUTPUT_ROOT / "samples_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[OK] Summary CSV saved: {summary_csv_path}")

    # Generate Markdown Documentation
    readme_path = OUTPUT_ROOT / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Hydrodynamic Entropy of Paintings — Curated Sample Showcase\n\n")
        f.write(
            "This folder contains **5 curated representative paintings per category (20 artworks total)** "
            "alongside their individual **Hydrodynamic Spectral Plots** and **Side-by-Side Comparison Figures** "
            "for use in presentations, reports, and slide decks.\n\n"
        )
        f.write("## Folder Structure\n\n")
        f.write("Each category subfolder (`classical/`, `modern/`, `picasso/`, `van_gogh/`) contains:\n")
        f.write("- `<id>_painting.jpg` : The original artwork in RGB\n")
        f.write("- `<id>_spectrum.png` : Log-log radial power spectrum $E(k)$ with Kolmogorov ($k^{-5/3}$) and Natural ($k^{-2}$) reference slopes\n")
        f.write("- `<id>_side_by_side.png` : Composite figure ready for PowerPoint (Artwork on Left, Spectrum on Right)\n\n")

        f.write("## Curated Samples Summary Table\n\n")
        f.write("| Category | ID | Artwork Title | $H_{\\rm norm}$ (Primary) | Spectral Slope $m$ | Pixel Entropy |\n")
        f.write("| :--- | :--- | :--- | :---: | :---: | :---: |\n")
        for r in summary_records:
            f.write(
                f"| **{r['category'].title()}** | `{r['short_id']}` | {r['artwork_title']} | "
                f"**{r['hydrodynamic_entropy_norm']:.3f}** | {r['spectral_slope_m']:.2f} | {r['pixel_spatial_entropy']:.2f} |\n"
            )

        f.write("\n## Category Averages Across Showcase Samples\n\n")
        cat_group = summary_df.groupby("category")[["hydrodynamic_entropy_norm", "spectral_slope_m", "pixel_spatial_entropy"]].mean()
        f.write(cat_group.to_markdown())
        f.write("\n\n---\n*IDC Computational Aesthetics Project — Verma & Chatterjee (Phys. Rev. Fluids 2022) Adaptation*\n")

    print(f"[OK] README.md saved: {readme_path}")
    print("\nAll 20 paintings, 20 plots, and 20 side-by-side slides created successfully!")


if __name__ == "__main__":
    main()
