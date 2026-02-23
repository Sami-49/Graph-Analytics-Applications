"""Create minimal sample dataset for testing."""
from pathlib import Path

def main():
    base = Path(__file__).parent / "sample_data"
    for subset in ["twitter15", "twitter16"]:
        (base / subset / "tree").mkdir(parents=True, exist_ok=True)
        labels = []
        for i in range(1, 31):
            eid = f"evt_{subset}_{i:03d}"
            labels.append(f"{eid} {'1' if i % 2 == 0 else '0'}")
            tree = base / subset / "tree" / f"{eid}.txt"
            edges = []
            n = 4 + (i % 8)
            for j in range(n):
                if j == 0:
                    edges.append(f"root u1")
                else:
                    edges.append(f"u{j} u{j+1}")
            tree.write_text("\n".join(edges), encoding="utf-8")
        (base / subset / "label.txt").write_text("\n".join(labels), encoding="utf-8")
    print(f"Sample data cree: {base.absolute()}")
    print('Dans config.yaml, mettez: dataset.root: "./sample_data"')

if __name__ == "__main__":
    main()
