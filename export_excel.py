"""
export_excel.py — Hydrodynamic Entropy of Paintings
Exports results to a formatted Excel workbook with:
  Sheet 1: Full results (all 400 images)
  Sheet 2: Per-category summary statistics
  Sheet 3: ANOVA results
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "spectral_entropy_results.csv"
XLSX_PATH = RESULTS_DIR / "spectral_entropy_results.xlsx"

CATEGORY_COLORS = {
    "classical": "D6E4FF",
    "modern":    "D6FFD6",
    "picasso":   "FFD6D6",
    "van_gogh":  "FFF4D6",
    "vangogh":   "FFF4D6",
    "other":     "E8D6FF",
}

HEADER_FILL  = PatternFill("solid", fgColor="2B4590")
HEADER_FONT  = Font(color="FFFFFF", bold=True)
BOLD_FONT    = Font(bold=True)
CENTER_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)
THIN_BORDER  = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"),  bottom=Side(style="thin"),
)


def _auto_width(ws):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            try:
                max_len = max(max_len, len(str(cell.value or "")))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max_len + 4, 30)


def _style_header_row(ws, row=1):
    for cell in ws[row]:
        cell.fill      = HEADER_FILL
        cell.font      = HEADER_FONT
        cell.alignment = CENTER_ALIGN
        cell.border    = THIN_BORDER


def build_full_results(writer, df):
    display_cols = ["category"]
    if "subcategory" in df.columns:
        display_cols.append("subcategory")
    display_cols.extend([
        "image_name",
        "hydrodynamic_entropy_norm",
        "spectral_slope",
        "spatial_entropy",
        "hydrodynamic_entropy",
        "hydrodynamic_entropy_max",
        "shell_hydrodynamic_entropy_norm",
    ])
    col_labels = {
        "category":                       "Category",
        "subcategory":                    "Subcategory / Tradition",
        "image_name":                     "Painting Name",
        "hydrodynamic_entropy_norm":      "H_norm (PRIMARY)",
        "spectral_slope":                 "Spectral Slope β",
        "spatial_entropy":                "Pixel Entropy (bits)",
        "hydrodynamic_entropy":           "H (bits)",
        "hydrodynamic_entropy_max":       "H_max (bits)",
        "shell_hydrodynamic_entropy_norm":"Shell H_norm",
    }
    out = df[display_cols].copy()
    out.rename(columns=col_labels, inplace=True)
    out.to_excel(writer, sheet_name="Full Results", index=False)

    ws = writer.sheets["Full Results"]
    _style_header_row(ws)

    # Category row colors
    for row_idx, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), start=0):
        cat_val = str(df["category"].iloc[row_idx]).lower().replace(" ", "_")
        fill_color = CATEGORY_COLORS.get(cat_val, "FFFFFF")
        fill = PatternFill("solid", fgColor=fill_color)
        for cell in row:
            cell.fill   = fill
            cell.border = THIN_BORDER
            cell.alignment = CENTER_ALIGN

    _auto_width(ws)
    ws.freeze_panes = "A2"


def build_summary(writer, df):
    metrics = [
        ("hydrodynamic_entropy_norm", "H_norm (PRIMARY)"),
        ("spectral_slope",            "Spectral Slope β"),
        ("spatial_entropy",           "Pixel Entropy (bits)"),
        ("hydrodynamic_entropy",      "H (bits)"),
    ]
    rows = []
    known_order = ["classical", "modern", "picasso", "van_gogh", "other"]
    cats = sorted(df["category"].str.lower().unique(), key=lambda c: known_order.index(c) if c in known_order else 999)
    for cat in cats:
        sub = df[df["category"].str.lower() == cat]
        display_name = cat.replace("_", " ").title()
        row = {"Category": display_name, "Count": len(sub)}
        for col, label in metrics:
            row[f"{label} — Mean"]   = round(sub[col].mean(), 4)
            row[f"{label} — Std"]    = round(sub[col].std(),  4)
            row[f"{label} — Min"]    = round(sub[col].min(),  4)
            row[f"{label} — Max"]    = round(sub[col].max(),  4)
            row[f"{label} — Median"] = round(sub[col].median(), 4)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_excel(writer, sheet_name="Category Summary", index=False)

    ws = writer.sheets["Category Summary"]
    _style_header_row(ws)
    for row_idx, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row), start=0):
        cat_val = summary_df["Category"].iloc[row_idx].lower().replace(" ", "_")
        fill_color = CATEGORY_COLORS.get(cat_val, "FFFFFF")
        fill = PatternFill("solid", fgColor=fill_color)
        for cell in row:
            cell.fill   = fill
            cell.border = THIN_BORDER
            cell.alignment = CENTER_ALIGN
    _auto_width(ws)
    ws.freeze_panes = "A2"


def build_anova(writer, df):
    categories = [g for _, g in df.groupby("category")]
    metrics = [
        ("hydrodynamic_entropy_norm",      "H_norm — Hydrodynamic Shell Entropy [PRIMARY]"),
        ("shell_hydrodynamic_entropy_norm","Shell H_norm"),
        ("spectral_slope",                  "Spectral Slope β"),
        ("spatial_entropy",                 "Pixel / Intensity Entropy [SECONDARY]"),
    ]
    rows = []
    for col, label in metrics:
        groups = [g[col].dropna().values for g in categories]
        F, p = stats.f_oneway(*groups)
        rows.append({
            "Metric": label,
            "F-statistic": round(F, 4),
            "p-value": f"{p:.3e}",
            "Significant (α=0.05)": "YES ✓" if p < 0.05 else "NO",
            "Interpretation": (
                "Strong difference between art styles"
                if p < 0.001 else
                "Moderate difference" if p < 0.05 else
                "No significant difference"
            ),
        })

    anova_df = pd.DataFrame(rows)
    anova_df.to_excel(writer, sheet_name="ANOVA Results", index=False)

    ws = writer.sheets["ANOVA Results"]
    _style_header_row(ws)
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.border    = THIN_BORDER
            cell.alignment = CENTER_ALIGN
    _auto_width(ws)


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print(f"Columns: {list(df.columns)}")

    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as writer:
        build_full_results(writer, df)
        build_summary(writer, df)
        build_anova(writer, df)

    print(f"\nExcel workbook saved → {XLSX_PATH}")
    print(f"  Sheet 1: Full Results   ({len(df)} rows, colour-coded by category)")
    print("  Sheet 2: Category Summary (mean / std / min / max / median)")
    print("  Sheet 3: ANOVA Results  (F-statistic + p-value for all metrics)")


if __name__ == "__main__":
    main()
