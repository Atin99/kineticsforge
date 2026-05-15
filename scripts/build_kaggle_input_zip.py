from pathlib import Path

from training.colab_kaggle.kaggle_input_builder import KaggleInputBuilder, write_one_cell


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "training" / "colab_kaggle" / "input_manifest" / "kineticsforge_v2_input.zip"
    manifest = KaggleInputBuilder(root).build(out)
    write_one_cell(root / "training" / "colab_kaggle" / "notebooks" / "KINETICSFORGE_V2_ONE_CELL.py", out.name)
    print({"zip": str(out), "files": manifest.file_count, "bytes": manifest.total_bytes})


if __name__ == "__main__":
    main()
