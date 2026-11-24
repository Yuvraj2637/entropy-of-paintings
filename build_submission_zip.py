import os
import zipfile
from pathlib import Path

def create_submission_zip():
    project_root = Path(".").resolve()
    zip_path = project_root / "Group_P09_Hydrodynamic_Entropy_Project.zip"
    
    # Files to include directly at root of zip
    root_files = [
        "README.md",
        "requirements.txt",
        "run_analysis.py",
        "generate_visualizations.py",
        "export_excel.py",
        "export_samples.py",
        "plot_single_scale_comparison.py",
        "print_summary.py",
    ]
    
    # Folders to include in full
    include_dirs = [
        "src",
        "sample_paintings_with_plots",
        "data/images",
    ]
    
    # Specific files from results/ to include (summary outputs, excel, key dashboard plots)
    results_dir = project_root / "results"
    
    print(f"Creating submission zip: {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add root files if they exist
        for fname in root_files:
            fpath = project_root / fname
            if fpath.exists():
                zf.write(fpath, arcname=f"Group_P09_Hydrodynamic_Entropy/{fname}")
                print(f"  + Added: {fname}")
        
        # Add included directories recursively
        for dir_rel in include_dirs:
            dp = project_root / dir_rel
            if dp.exists():
                for p in dp.rglob("*"):
                    if p.is_file() and not any(part.startswith(".") for part in p.parts) and "__pycache__" not in p.parts:
                        arcname = f"Group_P09_Hydrodynamic_Entropy/{p.relative_to(project_root).as_posix()}"
                        zf.write(p, arcname=arcname)
                print(f"  + Added folder: {dir_rel}")
        
        # Add results folder (excluding results/plots)
        if results_dir.exists():
            for p in results_dir.iterdir():
                if p.is_file():
                    arcname = f"Group_P09_Hydrodynamic_Entropy/results/{p.name}"
                    zf.write(p, arcname=arcname)
            print("  + Added: results/ (CSVs, Excel, Dashboard, Comparison Plots)")

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nDone! Submission zip created successfully:")
    print(f"  Path: {zip_path}")
    print(f"  Size: {zip_size_mb:.2f} MB")

if __name__ == "__main__":
    create_submission_zip()
