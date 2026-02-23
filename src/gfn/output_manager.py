from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
import hashlib
import json
import sys
from datetime import datetime


def sha256_file(path: Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class OutputPaths:
    run_root: Path
    manifest_json: Path
    config_snapshot_yaml: Path

    logs_dir: Path
    tables_dir: Path
    figures_dir: Path
    reports_dir: Path

    latest_file: Path
    index_md: Path


def _stable_json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False, default=str)


def _short_hash_from_config(config_dict: Mapping[str, Any]) -> str:
    s = json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:8]


def make_run_id(config_dict: Mapping[str, Any], now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{_short_hash_from_config(config_dict)}"


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class OutputManager:
    def __init__(self, outputs_root: Path, config_dict: Mapping[str, Any], run_id: Optional[str] = None):
        self.outputs_root = Path(outputs_root)
        self.outputs_root.mkdir(parents=True, exist_ok=True)

        self.config_dict = dict(config_dict)
        self.run_id = run_id or make_run_id(self.config_dict)
        self.run_root = self.outputs_root / "runs" / self.run_id

        self.logs_dir = self.run_root / "logs"
        self.tables_dir = self.run_root / "tables"
        self.figures_dir = self.run_root / "figures"
        self.reports_dir = self.run_root / "reports"

        self.manifest_json = self.run_root / "manifest.json"
        self.config_snapshot_yaml = self.run_root / "config.yaml"

        self.latest_file = self.outputs_root / "LATEST"
        self.index_md = self.outputs_root / "index.md"

    def create_layout(self) -> OutputPaths:
        _safe_mkdir(self.run_root)
        for d in [self.logs_dir, self.tables_dir, self.figures_dir, self.reports_dir]:
            _safe_mkdir(d)

        return OutputPaths(
            run_root=self.run_root,
            manifest_json=self.manifest_json,
            config_snapshot_yaml=self.config_snapshot_yaml,
            logs_dir=self.logs_dir,
            tables_dir=self.tables_dir,
            figures_dir=self.figures_dir,
            reports_dir=self.reports_dir,
            latest_file=self.latest_file,
            index_md=self.index_md,
        )

    def write_latest_pointer(self) -> None:
        self.latest_file.write_text(self.run_id, encoding="utf-8")

    def snapshot_config_yaml(self, config_yaml_text: str) -> None:
        self.config_snapshot_yaml.write_text(config_yaml_text, encoding="utf-8")

    def write_manifest(self, manifest: Mapping[str, Any]) -> Path:
        data = dict(manifest)
        data.setdefault("run_id", self.run_id)
        data.setdefault("timestamp", datetime.now().isoformat())
        data.setdefault("outputs_root", str(self.outputs_root))
        data.setdefault("run_root", str(self.run_root))
        self.manifest_json.write_text(_stable_json_dumps(data), encoding="utf-8")
        return self.manifest_json

    def update_global_index(self, run_summaries: Optional[List[Mapping[str, Any]]] = None) -> None:
        runs_dir = self.outputs_root / "runs"
        _safe_mkdir(runs_dir)
        run_ids = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()], reverse=True)

        lines: List[str] = ["# Outputs Index", "", f"Latest: `{self.latest_file.read_text(encoding='utf-8').strip()}`" if self.latest_file.exists() else "Latest: (none)", ""]
        lines.append("## Runs")
        lines.append("")

        summary_by_run: Dict[str, Mapping[str, Any]] = {}
        if run_summaries:
            for s in run_summaries:
                rid = str(s.get("run_id", ""))
                if rid:
                    summary_by_run[rid] = s

        for rid in run_ids:
            run_root = runs_dir / rid
            manifest_path = run_root / "manifest.json"
            lines.append(f"- `{rid}`")
            lines.append(f"  - path: `{run_root}`")
            if manifest_path.exists():
                lines.append(f"  - manifest: `{manifest_path}`")
            if rid in summary_by_run:
                s = summary_by_run[rid]
                for k, v in s.items():
                    if k == "run_id":
                        continue
                    lines.append(f"  - {k}: {v}")
        self.index_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def table_path(self, name: str) -> Path:
        return self.tables_dir / name

    def figure_path(self, name: str) -> Path:
        return self.figures_dir / name

    def report_path(self, name: str) -> Path:
        return self.reports_dir / name

    def list_artifacts(self) -> Dict[str, List[str]]:
        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(self.run_root)).replace("\\", "/")
            except Exception:
                return str(p)

        artifacts: Dict[str, List[str]] = {"tables": [], "figures": [], "reports": [], "logs": [], "other": []}
        for p in self.run_root.rglob("*"):
            if not p.is_file():
                continue
            if self.tables_dir in p.parents:
                artifacts["tables"].append(_rel(p))
            elif self.figures_dir in p.parents:
                artifacts["figures"].append(_rel(p))
            elif self.reports_dir in p.parents:
                artifacts["reports"].append(_rel(p))
            elif self.logs_dir in p.parents:
                artifacts["logs"].append(_rel(p))
            else:
                artifacts["other"].append(_rel(p))

        for k in artifacts:
            artifacts[k] = sorted(artifacts[k])
        return artifacts


def collect_environment_info(requirements_path: Optional[Path] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": sys.version.replace("\n", " "),
        "platform": sys.platform,
    }

    if requirements_path and Path(requirements_path).exists():
        try:
            txt = Path(requirements_path).read_text(encoding="utf-8")
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
            info["requirements"] = lines
        except Exception:
            info["requirements"] = []

    # Installed versions for key packages (best-effort, does not require pip).
    try:
        from importlib import metadata

        pkgs = [
            "numpy",
            "scipy",
            "scikit-learn",
            "networkx",
            "pandas",
            "matplotlib",
            "seaborn",
            "python-louvain",
            "xgboost",
            "pyyaml",
            "tqdm",
            "pypdf",
            "pymupdf",
        ]
        installed: Dict[str, str] = {}
        for name in pkgs:
            try:
                installed[name] = metadata.version(name)
            except Exception:
                continue
        info["installed_versions"] = installed
    except Exception:
        info["installed_versions"] = {}

    return info


def try_git_hash(repo_root: Path) -> str:
    import subprocess

    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"
