# Entropy of Paintings

**Course Project — Group P09**  
**Members:** Yuvraj Singh Chaudhary & Adarsh Vardhan  
**Theoretical Framework:** Adapted from *Verma & Chatterjee, Phys. Rev. Fluids 7, 114608 (2022)*

---

## 1. Project Overview

This project implements the Fourier hydrodynamic energy spectrum and shell entropy formulation to analyze statistical physics and spatial frequency distributions across 500 paintings across 5 distinct art traditions:
1. **Classical Realism** (100 paintings: Leonardo da Vinci, Raphael, Botticelli, etc.)
2. **Modern / Abstract** (100 paintings: Wassily Kandinsky, abstract expressionism)
3. **Pablo Picasso** (100 paintings: Cubism, Blue, Rose, and Surrealist periods)
4. **Vincent van Gogh** (100 paintings: Post-Impressionism, impasto brushwork)
5. **Other Traditions** (100 paintings: Raja Ravi Varma oil realism, Japanese Ukiyo-e woodblock prints by Hokusai/Hiroshige, and Symbolism/Romanticism by Munch, Klimt, and Friedrich)

---

## 2. Mathematical Formulation

### Total 2D Power Spectrum (RGB Energy Summation)
DC brightness is removed per color channel, and kinetic/Fourier power is summed across channels before binning:
$$\text{Power}(k_x, k_y) = \sum_{c \in \{R,G,B\}} |\mathcal{F}\{I_c(x, y) - \bar{I}_c\}|^2$$

### Isotropic Radial Shell Binning
Energy is integrated across concentric annular shells $S_k = \{ (k_x, k_y) \mid k - 0.5 \le \sqrt{k_x^2 + k_y^2} < k + 0.5 \}$:
$$E(k) = \sum_{(k_x, k_y) \in S_k} \text{Power}(k_x, k_y)$$

### Scale Probability & Normalized Hydrodynamic Entropy
$$p(k) = \frac{E(k)}{\sum_k E(k)}, \quad H = -\sum_k p(k) \log_2 p(k), \quad H_{\text{norm}} = \frac{H}{\log_2(N_{\text{shells}})}$$

### Physical Inertial Range Power-Law Fit
Spectral slope $\beta$ ($E(k) \sim k^\beta$) is fitted strictly on the physical inertial subrange:
$$k \in [6, 80]$$
This eliminates canvas boundary and low-frequency framing artifacts ($k < 6$) as well as craquelure, sensor, and digital noise floor ($k > 80$). Benchmarks include **Kolmogorov 2D turbulence** ($k^{-5/3} \approx k^{-1.67}$) and **Natural scale-invariance** ($k^{-2.00}$).

---

## 3. Results

[Read the IDC612 project report](idc612project_report.pdf).

The analysis covers 500 paintings. [The CSV](results/spectral_entropy_results.csv) contains the per-painting metrics, and [the Excel workbook](results/spectral_entropy_results.xlsx) includes the data, summary, and ANOVA.

- [Analysis dashboard](results/aesthetic_analysis_dashboard.png)
- [Aesthetic space map](results/2d_aesthetic_space_map.png)
- [Category spectra grid](results/category_spectra_grid_5panel.png)
- [Single-scale category comparison](results/single_scale_category_comparison.png)

[Download the 484 per-painting spectrum plots](https://github.com/Yuvraj2637/entropy-of-paintings/releases/download/results-v1/painting-spectra.zip). Extract the ZIP into `results/` to restore `results/plots/`.

The core spectral analysis is in [src/spectral_analysis.py](src/spectral_analysis.py). Place the source images under `data/images/` to reproduce the analysis. The input dataset and the course submission archives are distributed separately from these results.

## 4. How to Reproduce

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Arrange the artwork under `data/images/` using the `classical`, `modern`, `picasso`, `van_gogh`, and `other` categories. Then run the pipeline:
   ```bash
   python run_analysis.py
   ```
3. Generate Excel workbook and comparison figures:
   ```bash
   python export_excel.py
   python plot_single_scale_comparison.py
   ```
4. Print summary metrics to console:
   ```bash
   python print_summary.py
   ```
