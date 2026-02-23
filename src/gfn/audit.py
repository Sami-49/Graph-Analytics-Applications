"""Audit module: PDF parsing, table consistency checks, report generation."""
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)

_PYMUPDF_AVAILABLE: Optional[bool] = None


def _check_pymupdf() -> bool:
    """Check if PyMuPDF (fitz) is available."""
    global _PYMUPDF_AVAILABLE
    if _PYMUPDF_AVAILABLE is not None:
        return _PYMUPDF_AVAILABLE
    try:
        import fitz  # noqa: F401
        _PYMUPDF_AVAILABLE = True
    except ImportError:
        _PYMUPDF_AVAILABLE = False
    return _PYMUPDF_AVAILABLE


def audit_pdfs(pdf_paths: List[Path]) -> Dict[str, Any]:
    """
    Parse PDFs with PyMuPDF if available, else skip with note.
    Returns a dict with keys: parsed, skipped, errors, pymupdf_available.
    """
    result: Dict[str, Any] = {
        "parsed": [],
        "skipped": [],
        "errors": [],
        "pymupdf_available": False,
        "pypdf_available": False,
    }
    if not pdf_paths:
        return result

    result["pymupdf_available"] = _check_pymupdf()
    if not result["pymupdf_available"]:
        try:
            import pypdf  # noqa: F401
            result["pypdf_available"] = True
        except Exception:
            result["pypdf_available"] = False

    if not result["pymupdf_available"] and not result["pypdf_available"]:
        for p in pdf_paths:
            result["skipped"].append(str(p))
        return result

    fitz = None
    if result["pymupdf_available"]:
        import fitz as _fitz
        fitz = _fitz

    PdfReader = None
    if result["pypdf_available"]:
        from pypdf import PdfReader as _PdfReader
        PdfReader = _PdfReader

    for p in pdf_paths:
        p = Path(p)
        if not p.exists():
            result["errors"].append(f"File not found: {p}")
            continue
        if p.suffix.lower() != ".pdf":
            result["skipped"].append(str(p))
            continue
        try:
            pages = 0
            text_sample = ""
            full_text = ""

            if fitz is not None:
                doc = fitz.open(p)
                pages = len(doc)
                if pages > 0:
                    text_sample = doc[0].get_text()[:500].strip()
                full_text = "\n".join([(doc[i].get_text() or "") for i in range(min(pages, 3))])
                doc.close()
            elif PdfReader is not None:
                reader = PdfReader(str(p))
                pages = len(reader.pages)
                if pages > 0:
                    page0 = reader.pages[0]
                    text_sample = (page0.extract_text() or "")[:500].strip()
                full_text = "\n".join([
                    (reader.pages[i].extract_text() or "") for i in range(min(pages, 3))
                ])

            result["parsed"].append({
                "path": str(p),
                "pages": pages,
                "text_preview": text_sample[:200] + "..." if len(text_sample) > 200 else text_sample,
                "text_first_pages": full_text,
            })
        except Exception as e:
            result["errors"].append(f"{p}: {e}")
    return result


def _parse_discrimination_table_from_pdf_text(text: str) -> Dict[str, Dict[str, float]]:
    """
    Attempt to parse the discrimination table rendered as monospaced text in the PDF.

    Expected columns include (as printed by GraphMiningComparison):
    Model, Mean(Real), Mean(Fake), Difference, Cohen's d, Ranking AUC, KS Stat, KS p-val
    """
    if not text:
        return {}

    # Normalize whitespace while preserving line structure.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}

    # Find header line (support multiple formats).
    header_idx = None
    for i, ln in enumerate(lines[:250]):
        # Newer table format (seen in current PDFs)
        if "mean_real" in ln and "mean_fake" in ln and "ranking_auc" in ln:
            header_idx = i
            break
        # Legacy format (from GraphMiningComparison print)
        if "Mean(Real)" in ln and "Mean(Fake)" in ln and "Ranking" in ln:
            header_idx = i
            break
    if header_idx is None:
        return {}

    parsed: Dict[str, Dict[str, float]] = {}
    number_re = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

    # More permissive parsing than a strict regex because PDF extraction
    # may collapse multiple spaces or insert odd separators.
    for ln in lines[header_idx + 1:header_idx + 120]:
        if "_score" not in ln:
            continue

        toks = re.split(r"\s+", ln)
        if len(toks) < 8:
            continue
        model = toks[0]
        nums = []
        for t in toks[1:]:
            if re.fullmatch(number_re, t):
                nums.append(t)
        # Current PDFs have >= 12 numeric columns; legacy format has 7.
        if len(nums) < 7:
            continue

        try:
            row: Dict[str, float] = {
                "mean_real": float(nums[0]),
                "mean_fake": float(nums[1]),
                "mean_diff": float(nums[2]),
            }

            # If this is the new extended table layout:
            # mean_real mean_fake mean_diff std_real std_fake cohens_d ks_statistic ks_pvalue ranking_auc ...
            if len(nums) >= 9:
                row["std_real"] = float(nums[3])
                row["std_fake"] = float(nums[4])
                row["cohens_d"] = float(nums[5])
                row["ks_statistic"] = float(nums[6])
                row["ks_pvalue"] = float(nums[7])
                row["ranking_auc"] = float(nums[8])
            else:
                # Legacy compact format
                row["cohens_d"] = float(nums[3])
                row["ranking_auc"] = float(nums[4])
                row["ks_statistic"] = float(nums[5])
                row["ks_pvalue"] = float(nums[6])

            parsed[model] = row
        except Exception:
            continue
    return parsed


def audit_pdf_vs_tables(
    pdf_text: str,
    comparison_csv: Path,
    tolerance: float = 5e-5,
) -> Dict[str, Any]:
    """Compare PDF-extracted discrimination table against comparison_report_*.csv."""
    result: Dict[str, Any] = {
        "parsed_models": [],
        "contradictions": [],
        "notes": [],
    }

    parsed_tbl = _parse_discrimination_table_from_pdf_text(pdf_text)
    if not parsed_tbl:
        result["notes"].append("Could not parse discrimination table from PDF text.")
        return result

    try:
        import pandas as pd
        df = pd.read_csv(comparison_csv, index_col=0)
    except Exception as e:
        result["notes"].append(f"Failed to read {comparison_csv}: {e}")
        return result

    # Ensure numeric.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # PDFs round to ~4 decimals, so we treat equality after 4-dec rounding as consistent.
    def _close(v_pdf: float, v_csv: float) -> bool:
        if abs(v_pdf - v_csv) <= tolerance:
            return True
        try:
            return round(v_pdf, 4) == round(v_csv, 4)
        except Exception:
            return False

    for model, metrics in parsed_tbl.items():
        if model not in df.index:
            continue
        result["parsed_models"].append(model)

        for k in ["mean_real", "mean_fake", "mean_diff", "cohens_d", "ranking_auc", "ks_statistic", "ks_pvalue"]:
            if k not in metrics or k not in df.columns:
                continue
            v_pdf = float(metrics[k])
            v_csv = float(df.loc[model, k]) if df.loc[model, k] == df.loc[model, k] else None
            if v_csv is None:
                continue
            if not _close(v_pdf, v_csv):
                result["contradictions"].append(
                    {
                        "model": model,
                        "metric": k,
                        "pdf": v_pdf,
                        "csv": v_csv,
                        "abs_diff": abs(v_pdf - v_csv),
                    }
                )

    if not result["parsed_models"]:
        result["notes"].append("Parsed table did not match any model rows in comparison CSV.")

    return result


def audit_tables(tables_dir: Path) -> Dict[str, Any]:
    """
    Check model_scores CSVs for spectral_score constant/contradictions.
    Returns a dict with findings per file and cross-file consistency.
    """
    result: Dict[str, Any] = {
        "files_checked": [],
        "findings": [],
        "spectral_stats": {},
        "cross_file_issues": [],
        "impossible_cases": [],
    }
    tables_dir = Path(tables_dir)
    if not tables_dir.exists():
        result["findings"].append(f"Tables directory does not exist: {tables_dir}")
        return result

    model_score_files = list(tables_dir.glob("model_scores_*.csv"))
    if not model_score_files:
        # This is expected for diagnose-only runs.
        if (tables_dir / "dataset_overview.csv").exists():
            result["findings"].append("No model_scores_*.csv files found (expected for diagnose-only run)")
        else:
            result["findings"].append("No model_scores_*.csv files found")
        return result

    try:
        import pandas as pd
    except ImportError:
        result["findings"].append("pandas not available for table audit")
        return result

    spectral_means: Dict[str, float] = {}
    spectral_stds: Dict[str, float] = {}

    for f in model_score_files:
        try:
            df = pd.read_csv(f)
            result["files_checked"].append(str(f.name))
            if "spectral_score" not in df.columns:
                result["findings"].append(f"{f.name}: spectral_score column missing")
                continue

            vals = df["spectral_score"].dropna()
            if len(vals) == 0:
                result["findings"].append(f"{f.name}: spectral_score all NaN")
                continue

            std_val = float(vals.std())
            mean_val = float(vals.mean())
            spectral_means[f.stem] = mean_val
            spectral_stds[f.stem] = std_val

            if std_val < 1e-10:
                result["findings"].append(
                    f"{f.name}: spectral_score is constant (std={std_val:.2e})"
                )
            if vals.min() < 0 or vals.max() > 1:
                result["findings"].append(
                    f"{f.name}: spectral_score out of [0,1] (min={vals.min():.4f}, max={vals.max():.4f})"
                )
            result["spectral_stats"][f.stem] = {
                "mean": mean_val,
                "std": std_val,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "count": int(len(vals)),
            }

            # Impossible-ish case: constant score but ranking AUC not ~0.5.
            # This indicates downstream metric computation is inconsistent with score distribution.
            if std_val < 1e-10:
                subset = f.stem.replace("model_scores_", "")
                comp_path = tables_dir / f"comparison_report_{subset}.csv"
                if comp_path.exists():
                    try:
                        comp = pd.read_csv(comp_path, index_col=0)
                        if "ranking_auc" in comp.columns and "spectral_score" in comp.index:
                            auc = float(comp.loc["spectral_score", "ranking_auc"])
                            if abs(auc - 0.5) > 0.05:
                                result["impossible_cases"].append(
                                    f"{subset}: spectral_score is constant but ranking_auc={auc:.3f}"
                                )
                    except Exception:
                        pass
        except Exception as e:
            result["findings"].append(f"{f.name}: {e}")

    # Cross-file: large mean difference could indicate inconsistency
    if len(spectral_means) >= 2:
        means_list = list(spectral_means.values())
        max_diff = max(means_list) - min(means_list)
        if max_diff > 0.5:
            result["cross_file_issues"].append(
                f"Spectral score means differ widely across datasets (max_diff={max_diff:.3f})"
            )
    return result


def generate_audit_report(out_path: Path, pdf_result: Optional[Dict] = None, table_result: Optional[Dict] = None) -> Path:
    """
    Write outputs/reports/audit.md with findings.
    If pdf_result/table_result are None, will skip those sections.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Audit Report",
        "",
        f"Generated: {_timestamp()}",
        "",
    ]

    if pdf_result is not None:
        lines.extend([
            "## PDF Audit",
            "",
        ])
        if not pdf_result.get("pymupdf_available") and not pdf_result.get("pypdf_available"):
            lines.append("- No PDF text extractor available (install PyMuPDF or pypdf). PDFs skipped.")
            for s in pdf_result.get("skipped", []):
                lines.append(f"  - Skipped: {s}")
        else:
            lines.append(f"- Parsed: {len(pdf_result.get('parsed', []))} files")
            lines.append(f"- Errors: {len(pdf_result.get('errors', []))}")
            for p in pdf_result.get("parsed", []):
                lines.append(f"  - {p.get('path', '')}: {p.get('pages', 0)} pages")
            for e in pdf_result.get("errors", []):
                lines.append(f"  - Error: {e}")
        lines.append("")

        # Include a short excerpt to make parsing failures actionable.
        parsed = pdf_result.get("parsed", [])
        if parsed:
            lines.append("### Extracted text excerpts (first pages)")
            lines.append("")
            for p in parsed:
                excerpt = (p.get("text_first_pages", "") or "").strip().splitlines()
                excerpt = [ln for ln in excerpt if ln.strip()]
                lines.append(f"- **{Path(p.get('path','')).name}**:")
                for ln in excerpt[:12]:
                    lines.append(f"  - {ln[:200]}")
            lines.append("")

        if pdf_result.get("pdf_vs_table"):
            lines.extend([
                "### PDF vs Tables consistency",
                "",
            ])
            for subset, res in pdf_result["pdf_vs_table"].items():
                lines.append(f"- **{subset}**:")
                notes = res.get("notes", [])
                contradictions = res.get("contradictions", [])
                if notes:
                    for n in notes:
                        lines.append(f"  - Note: {n}")
                if contradictions:
                    lines.append(f"  - Contradictions: {len(contradictions)}")
                    for c in contradictions[:20]:
                        lines.append(
                            f"    - {c['model']}.{c['metric']}: pdf={c['pdf']:.6g}, csv={c['csv']:.6g} (|Δ|={c['abs_diff']:.3g})"
                        )
                else:
                    lines.append("  - Contradictions: 0")
            lines.append("")

    if table_result is not None:
        lines.extend([
            "## Table Audit (model_scores)",
            "",
        ])
        lines.append(f"Files checked: {', '.join(table_result.get('files_checked', []))}")
        lines.append("")
        if table_result.get("spectral_stats"):
            lines.append("### spectral_score statistics")
            for name, stats in table_result["spectral_stats"].items():
                lines.append(f"- **{name}**: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                            f"min={stats['min']:.4f}, max={stats['max']:.4f}, n={stats['count']}")
            lines.append("")
        if table_result.get("findings"):
            lines.append("### Findings")
            for f in table_result["findings"]:
                lines.append(f"- {f}")
            lines.append("")
        if table_result.get("cross_file_issues"):
            lines.append("### Cross-file issues")
            for i in table_result["cross_file_issues"]:
                lines.append(f"- {i}")
            lines.append("")

        if table_result.get("impossible_cases"):
            lines.append("### Impossible cases")
            for i in table_result["impossible_cases"]:
                lines.append(f"- {i}")
            lines.append("")
        if not table_result.get("findings") and not table_result.get("cross_file_issues"):
            lines.append("No issues found.")
            lines.append("")

    # Required narrative sections
    lines.extend([
        "## Diagnosis (what is wrong)",
        "",
        "- Reports must be generated from the same run's tables to avoid mixed snapshots.",
        "- This allows silent mixing of runs and contradictory metrics (e.g., `spectral_score` appearing as all-zeros in one PDF while non-zero in CSVs/other PDFs).",
        "",
        "## Most probable root causes", 
        "",
        "- No run isolation (`run_id`) and no enforced single source of truth for report inputs.",
        "- PDF generation is not currently wired into the main CLI pipeline; PDFs may be produced by older scripts/configurations.",
        "",
        "## What will be fixed", 
        "",
        "- Introduce `run_id` output directories and a `run_manifest.json` containing config hash, git hash, timestamp, and dataset counts.",
        "- Enforce that figures and PDFs are generated strictly from the exact CSV tables saved for the same run.",
        "- Add contradiction checks and refuse to build final reports if PDFs/tables mismatch beyond tolerance.",
        "",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _timestamp() -> str:
    """Return ISO timestamp."""
    from datetime import datetime
    return datetime.now().isoformat()


def run_audit(
    outputs_root: Path,
    subsets: Optional[List[str]] = None,
) -> Path:
    """Run full audit and write outputs/reports/audit.md under outputs_root."""
    outputs_root = Path(outputs_root)
    reports_dir = outputs_root / "reports"
    tables_dir = outputs_root / "tables"

    if subsets is None:
        subsets = []
        for p in reports_dir.glob("comparison_report_*.pdf"):
            name = p.stem.replace("comparison_report_", "")
            subsets.append(name)

    pdfs = [reports_dir / f"comparison_report_{s}.pdf" for s in subsets]
    pdfs = [p for p in pdfs if p.exists()]

    pdf_result = audit_pdfs(pdfs)
    table_result = audit_tables(tables_dir)

    # Compare PDFs vs tables where possible
    pdf_vs_table: Dict[str, Any] = {}
    for p in pdf_result.get("parsed", []):
        pdf_path = Path(p.get("path", ""))
        subset = pdf_path.stem.replace("comparison_report_", "")
        comp_csv = tables_dir / f"comparison_report_{subset}.csv"
        if not comp_csv.exists():
            pdf_vs_table[subset] = {"notes": [f"Missing comparison CSV: {comp_csv}"]}
            continue
        pdf_vs_table[subset] = audit_pdf_vs_tables(p.get("text_first_pages", ""), comp_csv)
    if pdf_vs_table:
        pdf_result["pdf_vs_table"] = pdf_vs_table

    out_path = reports_dir / "audit.md"
    return generate_audit_report(out_path, pdf_result=pdf_result, table_result=table_result)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Audit existing outputs (PDFs vs tables) and write audit.md")
    parser.add_argument("--outputs", default="outputs", help="Outputs root folder containing tables/ and reports/")
    parser.add_argument("--subsets", nargs="*", default=None, help="Subset names (e.g., twitter15 twitter16)")
    parser.add_argument("--dump-text", action="store_true", help="Dump extracted PDF text into outputs/reports/")
    args = parser.parse_args()

    out_root = Path(args.outputs)
    out = run_audit(out_root, subsets=args.subsets)
    if args.dump_text:
        reports_dir = out_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        pdfs = audit_pdfs(list(reports_dir.glob("comparison_report_*.pdf")))
        for item in pdfs.get("parsed", []):
            p = Path(item.get("path", ""))
            txt = item.get("text_first_pages", "") or ""
            (reports_dir / f"{p.stem}__text.txt").write_text(txt, encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
