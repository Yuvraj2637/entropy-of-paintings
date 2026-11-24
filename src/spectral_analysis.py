"""
spectral_analysis.py — Hydrodynamic Entropy of Paintings
Implementation based on Verma & Chatterjee (Phys. Rev. Fluids 7, 114608, 2022).

Mathematical Formulation:
  1. Shell Energy Spectrum:    E(k) = sum_{|k'|=k} Power(k')
  2. Scale Probability:        p(k) = E(k) / sum_k E(k)
  3. Shell Entropy:            S    = - sum_k p(k) * log2(p(k))
  4. Normalized Entropy:       S_norm = S / log2(N_shells)

Color Channel Treatment:
  Total Fourier power is summed across R, G, B channels first:
    Power(kx, ky) = Power_R(kx, ky) + Power_G(kx, ky) + Power_B(kx, ky)
  This ensures energy-weighted contribution and prevents noise amplification
  in monochromatic or chromatic-dominant paintings.
"""

import time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def safe_savefig(fig, path, **kwargs):
    """Save matplotlib figure with retry logic against OneDrive/OS file locks."""
    for attempt in range(5):
        try:
            fig.savefig(path, **kwargs)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.5)


# ============================================================
# STEP 1: LOAD IMAGE
# ============================================================
def load_image(image_path: str) -> np.ndarray:
    """Load image as uint8 RGB NumPy array of shape (H, W, 3)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.asarray(img)


# ============================================================
# STEP 2: TOTAL 2D POWER SPECTRUM (RGB SUMMED FIRST)
# ============================================================
def compute_total_power_spectrum(image_rgb: np.ndarray) -> np.ndarray:
    """
    Computes 2D Fourier power spectrum summed across R, G, B channels.
    DC component (mean brightness) is subtracted per channel to isolate
    brushstrokes, edges, and texture variations.
    
    Total Power(kx, ky) = Power_R(kx, ky) + Power_G(kx, ky) + Power_B(kx, ky)
    """
    H, W, _ = image_rgb.shape
    total_power = np.zeros((H, W), dtype=np.float64)

    for c in range(3):
        channel = image_rgb[:, :, c].astype(np.float64)
        channel_ac = channel - np.mean(channel)  # Remove DC brightness spike
        fft_c = np.fft.fft2(channel_ac)
        total_power += np.abs(fft_c) ** 2       # Sum physical energy across channels

    return total_power


# ============================================================
# STEP 3: CONCENTRIC CIRCULAR SHELLS & HYDRODYNAMIC ENTROPY
# Radial shell integration:
#   1. E(k) = sum_{|k'|=k} Power(k')
#   2. p(k) = E(k) / sum_k E(k)
#   3. S    = - sum_k p(k) * log2(p(k))
# ============================================================
def compute_shell_energy_and_entropy(total_power: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Bins 2D Fourier modes (kx, ky) into concentric radial shells of radius k.
    
    Returns:
        k_values: 1D array of radial spatial frequencies k = 1, 2, ..., k_max
        E_k:      1D array of total energy in each shell k
        S:        Absolute hydrodynamic shell entropy in bits
        S_norm:   Normalized hydrodynamic entropy S / log2(N_shells) in [0, 1]
    """
    H, W = total_power.shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.round(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)).astype(int)

    # Center zero-frequency DC mode at (cy, cx)
    power_shifted = np.fft.fftshift(total_power)
    max_k = min(H, W) // 2  # Isotropic Nyquist circle (avoids rectangular edge distortion)

    r_flat = r.ravel()
    power_flat = power_shifted.ravel()

    # 1. Shell Energy: E(k) = sum of power modes falling inside shell k
    shell_energy_bincount = np.bincount(r_flat, weights=power_flat)
    valid_k = np.arange(len(shell_energy_bincount))
    mask = (shell_energy_bincount > 0) & (valid_k >= 1) & (valid_k <= max_k)

    k_values = valid_k[mask]
    E_k = shell_energy_bincount[mask]

    total_energy = np.sum(E_k)
    if total_energy <= 0 or len(E_k) <= 1:
        return k_values, E_k, 0.0, 0.0

    # 2. Scale Probability Distribution: p(k) = E(k) / sum(E(k))
    p_k = E_k / total_energy

    # 3. Hydrodynamic Shell Entropy: S = - sum(p(k) * log2(p(k)))
    p_k_nonzero = p_k[p_k > 0]
    S = float(-np.sum(p_k_nonzero * np.log2(p_k_nonzero)))

    # Normalized entropy S_norm in [0, 1]
    S_max = np.log2(len(k_values))
    S_norm = float(S / S_max) if S_max > 0 else 0.0

    return k_values, E_k, S, S_norm


# ============================================================
# STEP 4: FIT SPECTRAL SLOPE & PLOT WITH BENCHMARKS
# ============================================================
def save_radial_plot_and_get_slope(
    k_values: np.ndarray,
    E_k: np.ndarray,
    image_name: str,
    output_dir: Path,
    S_norm: float,
    S_bits: float,
) -> float:
    """
    Fits power-law slope m on log10(E(k)) vs log10(k): E(k) ~ k^m.
    Saves plot with Kolmogorov 2D turbulence (k^-5/3) and Natural scene (k^-2) benchmarks.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{Path(image_name).stem}_spectrum.png"

    # Fit power-law on the physical inertial range [6, 80]
    # (excludes k < 6 canvas boundary/framing effects and k > 80 sensor/craquelure noise floor)
    k_max = k_values[-1]
    fit_mask = (k_values >= 6) & (k_values <= min(80, k_max))
    
    log_k_fit = np.log10(k_values[fit_mask])
    log_E_fit = np.log10(E_k[fit_mask])

    # Linear fit in log-log space over inertial range
    slope, intercept = np.polyfit(log_k_fit, log_E_fit, deg=1)
    fitted_line = 10 ** (intercept + slope * np.log10(k_values))

    # Anchored benchmark lines at center of the inertial range (k = 25)
    k_mid = 25
    E_mid = 10 ** (intercept + slope * np.log10(k_mid))
    kolmogorov_line = E_mid * (k_values / k_mid) ** (-5.0 / 3.0)
    natural_line = E_mid * (k_values / k_mid) ** (-2.0)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.plot(k_values, E_k, color="#111111", linewidth=2.0, label=r"Total Energy Spectrum $E(k)$")
    ax.plot(k_values, fitted_line, color="#ff7f00", linewidth=2.2, label=f"Fitted Slope (m = {slope:.2f})")
    ax.plot(k_values, kolmogorov_line, color="#984ea3", linestyle="--", linewidth=1.6, alpha=0.85,
            label=r"Kolmogorov Turbulence ($k^{-5/3}$)")
    ax.plot(k_values, natural_line, color="#555555", linestyle=":", linewidth=1.6, alpha=0.85,
            label=r"Natural Scale-Invariance ($k^{-2}$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber / Spatial Frequency (k)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Shell Energy E(k)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Hydrodynamic Spectrum: {image_name}\n"
        f"Slope m = {slope:.2f} | Shell Entropy H_norm = {S_norm:.3f} ({S_bits:.2f} bits)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="lower left", framealpha=0.92)
    ax.grid(True, which="both", ls="--", alpha=0.35)

    plt.tight_layout()
    safe_savefig(fig, save_path, dpi=300)
    plt.close(fig)

    return float(slope)


# ============================================================
# STEP 5: PIXEL SPATIAL ENTROPY (SECONDARY BASELINE)
# ============================================================
def compute_spatial_entropy(image_rgb: np.ndarray) -> float:
    """
    Standard Shannon entropy of pixel intensity histogram (0 to 8 bits).
    Kept as a secondary, non-hydrodynamic baseline metric.
    """
    gray = (0.299 * image_rgb[:, :, 0] + 0.587 * image_rgb[:, :, 1] + 0.114 * image_rgb[:, :, 2]).astype(np.uint8)
    counts, _ = np.histogram(gray, bins=256, range=(0, 256))
    probabilities = counts[counts > 0] / gray.size
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)


# ============================================================
# MAIN ANALYSIS PIPELINE FOR A SINGLE IMAGE
# ============================================================
def analyze_image(image_path: str) -> Dict[str, float]:
    """
    Analyzes an artwork using the Hydrodynamic Shell Entropy formulation:
      1. Computes Total 2D Power Spectrum (summed across R, G, B channels).
      2. Performs isotropic radial shell binning -> E(k), p(k), and S.
      3. Fits power-law spectral slope m and saves radial plot.
      4. Computes baseline spatial pixel intensity entropy.
    """
    image_rgb = load_image(image_path)

    # 1. Total 2D power spectrum (summed across R, G, B first)
    total_power = compute_total_power_spectrum(image_rgb)

    # 2. Concentric shell binning and entropy calculation
    k_vals, E_k, S_bits, S_norm = compute_shell_energy_and_entropy(total_power)

    # 3. Fit slope & save spectrum plot
    plot_dir = Path("results/plots")
    slope = save_radial_plot_and_get_slope(
        k_vals, E_k, Path(image_path).name, plot_dir, S_norm, S_bits
    )

    # 4. Secondary pixel entropy
    pixel_entropy = compute_spatial_entropy(image_rgb)

    return {
        # Primary hydrodynamic shell entropy metrics
        "hydrodynamic_entropy": S_bits,
        "hydrodynamic_entropy_norm": S_norm,
        "hydrodynamic_entropy_max": float(np.log2(len(k_vals))) if len(k_vals) > 0 else 0.0,
        "shell_hydrodynamic_entropy": S_bits,
        "shell_hydrodynamic_entropy_norm": S_norm,
        "spectral_slope": slope,
        "spatial_entropy": pixel_entropy,
        # Backward compatibility aliases
        "entropy_R": S_norm,
        "entropy_G": S_norm,
        "entropy_B": S_norm,
        "entropy_RGB_mean": S_norm,
    }
