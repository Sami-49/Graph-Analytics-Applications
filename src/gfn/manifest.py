"""Manifest module: create run manifest with config_hash, git_hash, timestamp."""
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
import json
from datetime import datetime


def _config_hash(config: Any) -> str:
    """Compute hash of config dict for reproducibility."""
    try:
        d = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        s = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


def _git_hash(repo_root: Optional[Path] = None) -> str:
    """Get current git commit hash if available."""
    import subprocess
    try:
        root = Path(repo_root) if repo_root else None
        if root is None:
            p = Path(__file__).resolve().parent
            while p != p.parent:
                if (p / ".git").exists():
                    root = p
                    break
                p = p.parent
            root = root or Path(__file__).resolve().parents[2]
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=root,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def create_run_manifest(
    config: Any,
    dataset_counts: Dict[str, int],
    out_path: Path,
    repo_root: Optional[Path] = None,
) -> Path:
    """
    Write run_manifest.json with config_hash, git_hash, timestamp.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "config_hash": _config_hash(config),
        "git_hash": _git_hash(repo_root),
        "timestamp": datetime.now().isoformat(),
        "dataset_counts": dataset_counts,
    }
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to write manifest to {out_path}: {e}") from e
    return out_path
