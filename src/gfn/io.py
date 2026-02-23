"""Dataset I/O: parsing labels and cascade graphs."""
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


class LabelParser:
    """Parse label files with multiple format support."""

    @staticmethod
    def parse_labels(label_file: Path) -> Dict[str, int]:
        """Parse label file: event_id -> label (0 or 1)."""
        labels = {}
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # Support multiple formats:
                # Format 1: "event_id label" (space/tab separated)
                # Format 2: "label:event_id" (colon separated)
                
                if ":" in line:
                    # Format: label:event_id
                    parts = line.split(":")
                    if len(parts) == 2:
                        label_str = parts[0].lower().strip()
                        event_id = parts[1].strip()
                    else:
                        continue
                else:
                    # Format: event_id label
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    event_id = parts[0]
                    label_str = parts[1].lower()

                # Map label strings to 0/1
                if label_str in ["1", "true", "rumor", "fake", "false"]:
                    labels[event_id] = 1
                elif label_str in ["0", "non-rumor", "real"]:
                    labels[event_id] = 0
                elif label_str in ["unverified"]:
                    # Optional: treat unverified as rumor (1)
                    labels[event_id] = 1
        
        return labels


class CascadeLoader:
    """Load cascade graphs from tree files."""

    @staticmethod
    def parse_tree_file(tree_file: Path) -> List[Tuple[str, str]]:
        """Parse cascade file and extract edges."""
        edges = []
        with open(tree_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Try multiple separators
                for sep in [" ", "\t", ","]:
                    if sep in line:
                        parts = [p.strip() for p in line.split(sep)]
                        if len(parts) >= 2:
                            try:
                                src, tgt = parts[0], parts[1]
                                edges.append((src, tgt))
                            except (IndexError, ValueError):
                                pass
                        break
        return edges

    @staticmethod
    def load_all_subsets(
        paths, config
    ) -> Dict[str, Tuple[pd.DataFrame, Dict[str, int]]]:
        """Load all cascades from all subsets."""
        result = {}
        subsets = paths.find_subsets()

        for subset in subsets:
            labels = LabelParser.parse_labels(paths.find_label_file(subset))
            event_dirs = paths.list_event_dirs(subset)

            cascades = []
            skipped = defaultdict(int)

            for event_id in tqdm(
                event_dirs,
                desc=f"Loading {subset}",
                total=len(event_dirs),
            ):
                if config.dataset.max_events and len(cascades) >= config.dataset.max_events:
                    break

                try:
                    tree_file = paths.find_event_file(subset, event_id)
                    edges = CascadeLoader.parse_tree_file(tree_file)

                    if len(edges) < config.dataset.min_nodes - 1:
                        skipped["too_small"] += 1
                        continue

                    cascades.append({
                        "event_id": event_id,
                        "n_edges": len(edges),
                        "label": labels.get(event_id, -1),
                    })
                except Exception as e:
                    skipped["parse_error"] += 1

            df = pd.DataFrame(cascades)
            result[subset] = (df, labels)
            print(f"  {subset}: loaded {len(cascades)}, skipped {dict(skipped)}")

        return result
