import argparse
import json
from pathlib import Path


def require_chgnet():
    try:
        from chgnet.model import CHGNet
    except Exception as exc:
        raise SystemExit("Install chgnet on Kaggle/Colab before running: pip install chgnet") from exc
    return CHGNet


def load_graph_index(index_path: Path):
    try:
        import pandas as pd
    except Exception as exc:
        raise SystemExit("pandas is required to read the normalized crystal graph index") from exc
    if not index_path.exists():
        raise SystemExit(f"Missing crystal graph index: {index_path}")
    return pd.read_parquet(index_path)


def run_chgnet_finetune(index_path: Path, out_dir: Path, epochs: int = 50, batch_size: int = 16) -> dict:
    CHGNet = require_chgnet()
    table = load_graph_index(index_path)
    usable = table.dropna(subset=["formation_energy_per_atom"])
    if usable.empty:
        raise SystemExit("No formation-energy labels found. Harvest and normalize Materials Project data first.")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = CHGNet.load()
    manifest = {
        "status": "prepared",
        "model": "CHGNet pretrained Materials Project checkpoint",
        "records": int(len(usable)),
        "epochs_requested": int(epochs),
        "batch_size": int(batch_size),
        "note": "This entry point intentionally avoids local cold-start GNN training. Run fine-tuning on Kaggle/Colab with chgnet installed.",
        "output_dir": str(out_dir),
    }
    (out_dir / "chgnet_finetune_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _ = model
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/real/crystal_structures/normalized/crystal_graph_index.parquet")
    parser.add_argument("--out-dir", default="checkpoints/chgnet_naion")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    manifest = run_chgnet_finetune(Path(args.index), Path(args.out_dir), epochs=args.epochs, batch_size=args.batch_size)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
