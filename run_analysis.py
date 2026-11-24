from pathlib import Path
import pandas as pd
from src.spectral_analysis import analyze_image

# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_ROOT = Path("data/images")
RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "spectral_entropy_results.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_category(image_path: Path) -> str:
    """Category is the top-level folder directly under data/images."""
    rel = image_path.relative_to(IMAGE_ROOT)
    return rel.parts[0]


def get_subcategory(image_path: Path) -> str:
    """Subcategory (e.g. raja_ravi_varma, japanese_ukiyo_e, or top-level category)."""
    rel = image_path.relative_to(IMAGE_ROOT)
    if len(rel.parts) > 2:
        return rel.parts[1]
    return rel.parts[0]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Find all images
    # --------------------------------------------------------
    image_files = sorted([
        path for path in IMAGE_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print("No images found.\nPut images inside data/images/<category>/")
        return

    print(f"Found {len(image_files)} image(s).\n" + "-" * 60)

    results = []
    failed_images = []

    # --------------------------------------------------------
    # Analyze every image
    # --------------------------------------------------------
    for i, image_path in enumerate(image_files, start=1):
        safe_path = str(image_path).encode('ascii', errors='replace').decode('ascii')
        print(f"[{i}/{len(image_files)}] Analyzing: {safe_path}")

        try:
            analysis = analyze_image(str(image_path))
            result = {
                "category": get_category(image_path),
                "subcategory": get_subcategory(image_path),
                "image_name": image_path.name,
                "image_path": str(image_path),
                "hydrodynamic_entropy": analysis["hydrodynamic_entropy"],
                "hydrodynamic_entropy_norm": analysis["hydrodynamic_entropy_norm"],
                "hydrodynamic_entropy_max": analysis["hydrodynamic_entropy_max"],
                "shell_hydrodynamic_entropy": analysis["shell_hydrodynamic_entropy"],
                "shell_hydrodynamic_entropy_norm": analysis["shell_hydrodynamic_entropy_norm"],
                "entropy_R": analysis["entropy_R"],
                "entropy_G": analysis["entropy_G"],
                "entropy_B": analysis["entropy_B"],
                "entropy_RGB_mean": analysis["entropy_RGB_mean"],
                "spatial_entropy": analysis["spatial_entropy"],
                "spectral_slope": analysis["spectral_slope"],
            }
            results.append(result)

        except Exception as exc:
            print(f"ERROR: {image_path}\nReason: {exc}")
            failed_images.append(str(image_path))

    if not results:
        print("No images were successfully analyzed.")
        return

    # --------------------------------------------------------
    # Create DataFrame & Save
    # --------------------------------------------------------
    df = pd.DataFrame(results)

    try:
        df.to_csv(OUTPUT_FILE, index=False)
    except PermissionError:
        print("\nPermission denied while saving CSV. Close it if open in another app.")
        return

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE: HYDRODYNAMIC ENTROPY OF PAINTINGS")
    print("=" * 60)
    print(f"Successfully analyzed: {len(results)}")
    print(f"Failed: {len(failed_images)}")
    print(f"Results saved to: {OUTPUT_FILE}\n")

    print("\nAverage Hydrodynamic Results by Category (Verma & Chatterjee 2022):")
    # Group by category and calculate the mean for hydrodynamic entropy, slope, and spatial entropy
    summary_cols = [
        "hydrodynamic_entropy",
        "hydrodynamic_entropy_norm",
        "shell_hydrodynamic_entropy",
        "spectral_slope",
        "spatial_entropy",
    ]
    average_df = df.groupby("category")[summary_cols].mean()
    print(average_df)

    # --------------------------------------------------------
    # Generate Visual Dashboards
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("GENERATING AESTHETIC VISUALIZATIONS & STATISTICAL TESTS")
    print("=" * 60)
    try:
        from generate_visualizations import main as generate_all_visuals
        generate_all_visuals()
    except Exception as e:
        print(f"Visualization generation note: {e}")


if __name__ == "__main__":
    main()