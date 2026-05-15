import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data.dataset_contracts import ensure_dir, sha256_file, utc_now, write_json


def load_structure(record: Dict[str, object]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    structure = record.get("structure") or {}
    sites = structure.get("sites") or []
    lattice = np.asarray((structure.get("lattice") or {}).get("matrix") or np.eye(3), dtype=float)
    species = []
    frac = []
    for site in sites:
        label = ""
        sp = site.get("species") or []
        if sp:
            label = str(sp[0].get("element") or sp[0].get("label") or "")
        species.append(label)
        frac.append(site.get("abc") or site.get("frac_coords") or [0.0, 0.0, 0.0])
    return species, np.asarray(frac, dtype=float), lattice


def graph_from_structure(species: List[str], frac: np.ndarray, lattice: np.ndarray, cutoff_A: float = 3.2) -> Dict[str, object]:
    cart = frac @ lattice
    edges = []
    distances = []
    for i in range(len(species)):
        for j in range(len(species)):
            if i == j:
                continue
            d = float(np.linalg.norm(cart[i] - cart[j]))
            if d <= cutoff_A:
                edges.append([i, j])
                distances.append(d)
    return {
        "species": species,
        "frac_coords": frac.tolist(),
        "cart_coords": cart.tolist(),
        "edge_index": edges,
        "edge_distance_A": distances,
        "cutoff_A": cutoff_A,
    }


def normalize_directory(in_dir: Path, out_dir: Path, cutoff_A: float = 3.2) -> Dict[str, object]:
    out_dir = ensure_dir(out_dir)
    rows = []
    for path in sorted(in_dir.glob("*.json")):
        if path.name == "crystal_structure_manifest.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        species, frac, lattice = load_structure(record)
        graph = graph_from_structure(species, frac, lattice, cutoff_A=cutoff_A)
        material_id = str(record.get("material_id") or path.stem)
        out_path = out_dir / f"{material_id}_graph.json"
        payload = {
            "material_id": material_id,
            "formula": record.get("formula_pretty", ""),
            "formation_energy_per_atom": record.get("formation_energy_per_atom"),
            "energy_above_hull": record.get("energy_above_hull"),
            "band_gap": record.get("band_gap"),
            "graph": graph,
            "source_path": str(path),
            "source_sha256": sha256_file(path),
            "normalized_at": utc_now(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        rows.append({
            "material_id": material_id,
            "formula": payload["formula"],
            "n_atoms": len(species),
            "n_edges": len(graph["edge_index"]),
            "formation_energy_per_atom": payload["formation_energy_per_atom"],
            "energy_above_hull": payload["energy_above_hull"],
            "band_gap": payload["band_gap"],
            "graph_path": str(out_path),
            "graph_sha256": sha256_file(out_path),
        })
    table = pd.DataFrame(rows)
    table_path = out_dir / "crystal_graph_index.parquet"
    if not table.empty:
        table.to_parquet(table_path, index=False)
    manifest = {
        "created_at": utc_now(),
        "records": len(rows),
        "cutoff_A": cutoff_A,
        "paths": {"index": str(table_path), "graphs": str(out_dir)},
    }
    write_json(out_dir / "crystal_graph_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="data/real/crystal_structures/materials_project")
    parser.add_argument("--out-dir", default="data/real/crystal_structures/normalized")
    parser.add_argument("--cutoff-A", type=float, default=3.2)
    args = parser.parse_args()
    manifest = normalize_directory(Path(args.in_dir), Path(args.out_dir), cutoff_A=args.cutoff_A)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
