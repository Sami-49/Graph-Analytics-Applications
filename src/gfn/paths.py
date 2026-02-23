"""Dataset path discovery and validation."""
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict


class DatasetPaths:
    """Automatic discovery of dataset structure."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")

    def find_subsets(self) -> List[str]:
        """Find twitter15, twitter16 subdirectories."""
        subsets = []
        for d in self.root.iterdir():
            if d.is_dir() and d.name.lower() in ["twitter15", "twitter16"]:
                subsets.append(d.name)
        return sorted(subsets)

    def find_label_file(self, subset: str) -> Path:
        """Find label.txt in subset directory."""
        subset_dir = self.root / subset
        for fname in ["label.txt", "labels.txt"]:
            fpath = subset_dir / fname
            if fpath.exists():
                return fpath
        raise FileNotFoundError(f"No label file found in {subset_dir}")

    def list_event_dirs(self, subset: str) -> List[str]:
        """List event IDs (from tree/*.txt files or subdirectories)."""
        tree_dir = self.root / subset / "tree"
        if not tree_dir.exists():
            raise FileNotFoundError(f"tree/ directory not found in {subset}")
        
        event_ids = set()
        
        # Try direct files (twitter15/tree/*.txt)
        for f in tree_dir.iterdir():
            if f.is_file() and f.suffix == ".txt":
                # Filename is event ID
                event_ids.add(f.stem)
        
        # Try subdirectories (fallback)
        for d in tree_dir.iterdir():
            if d.is_dir():
                event_ids.add(d.name)
        
        return sorted(list(event_ids))

    def find_event_file(self, subset: str, event_id: str) -> Path:
        """Find cascade file for a given event."""
        tree_dir = self.root / subset / "tree"
        
        # Try direct file: tree/{event_id}.txt
        for fname in [f"{event_id}.txt", "tree.txt", "graph.txt", "cascade.txt", "edge.txt"]:
            fpath = tree_dir / fname
            if fpath.exists():
                return fpath
        
        # Try subdirectory
        event_subdir = tree_dir / event_id
        if event_subdir.exists():
            for fname in ["tree.txt", "graph.txt", "cascade.txt", "edge.txt"]:
                fpath = event_subdir / fname
                if fpath.exists():
                    return fpath
        
        raise FileNotFoundError(f"No cascade file found for {subset}/{event_id}")

    def diagnose(self) -> dict:
        """Print dataset diagnostics."""
        subsets = self.find_subsets()
        result: dict = {"subsets_detected": subsets}
        for subset in subsets:
            try:
                label_file = self.find_label_file(subset)
                event_dirs = self.list_event_dirs(subset)
                from .io import LabelParser, CascadeLoader
                labels = LabelParser.parse_labels(label_file)
                n_labeled = sum(1 for eid in event_dirs if labels.get(eid, -1) in [0, 1])
                di = {
                    "label_file": str(label_file),
                    "n_events": len(event_dirs),
                    "n_labeled": n_labeled,
                    "sample_events": event_dirs[:3] if event_dirs else [],
                }
                if event_dirs:
                    tree_file = self.find_event_file(subset, event_dirs[0])
                    edges = CascadeLoader.parse_tree_file(tree_file)
                    di["example_event_file"] = str(tree_file)
                    di["example_edges_parsed"] = len(edges)
                result[subset] = di
            except Exception as e:
                result[subset] = {"error": str(e)}
        return result
