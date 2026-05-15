import argparse
import json
from pathlib import Path

import numpy as np

from modules.bms.topology_variants import SyntheticPackGraphGenerator


class SyntheticBMSDataGenerator:
    def __init__(self, out_dir, seed=20260511):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.RandomState(seed)
        self.graphs = SyntheticPackGraphGenerator(seed=seed)

    def generate_instance(self, idx, duration_steps=900, family="mixed", inject_failure=True):
        graph = self.graphs.sample_topology(family=family)
        n_cells = int(graph["n_cells"].item())
        time = np.arange(duration_steps, dtype=np.float32)
        phase = 2.0 * np.pi * time / max(duration_steps, 1)
        I_pack = (18.0 * np.sin(phase * 3.0) + 5.0 * np.sin(phase * 17.0)).astype(np.float32)
        I_cells = I_pack[:, None] / max(n_cells, 1)
        capacity = graph["capacity_multiplier"].numpy().reshape(1, n_cells)
        T_amb = (308.0 + 8.0 * np.sin(phase - 0.7)).astype(np.float32)
        V_cells = 3.35 + 0.12 * capacity + 0.025 * self.rng.normal(size=(duration_steps, n_cells))
        R_int = 0.028 / np.clip(capacity, 0.8, 1.2) + 0.0015 * self.rng.normal(size=(duration_steps, n_cells))
        T_cells = T_amb[:, None] + 7.0 * np.abs(I_cells) * R_int + self.rng.normal(0.0, 0.4, size=(duration_steps, n_cells))
        risk = 1.0 / (1.0 + np.exp(-(T_cells - 346.0) / 6.0 - (R_int - 0.035) * 60.0))
        failure_step = np.full(n_cells, duration_steps, dtype=np.int32)
        if inject_failure:
            cell = int(graph["failure_cell"].item())
            onset = min(int(graph["failure_onset_step"].item()), duration_steps - 2)
            ramp = np.linspace(0.0, 1.0, duration_steps - onset)
            if int(graph["failure_mode"].item()) == 0:
                T_cells[onset:, cell] += 70.0 * ramp
                risk[onset:, cell] = np.maximum(risk[onset:, cell], 0.45 + 0.55 * ramp)
            elif int(graph["failure_mode"].item()) == 1:
                R_int[onset:, cell] += 0.08 * ramp
                risk[onset:, cell] = np.maximum(risk[onset:, cell], 0.35 + 0.65 * ramp)
            else:
                T_cells[onset:, cell] += 35.0 * ramp
                R_int[onset:, cell] += 0.045 * ramp
                risk[onset:, cell] = np.maximum(risk[onset:, cell], 0.40 + 0.60 * ramp)
            failure_step[cell] = onset
        P_dendrite = np.clip((R_int - 0.03) * 4.0, 0.0, 1.0).astype(np.float32)
        L_sei = np.clip((R_int - 0.02) * 1e-3, 1e-10, 5e-2).astype(np.float32)
        path = self.out_dir / f"bms_pack_graph_{idx:05d}.npz"
        compat_path = self.out_dir / f"bms_{idx:05d}.npz"
        np.savez_compressed(
            path,
            time=time,
            I_pack=I_pack,
            V_cells=V_cells.astype(np.float32),
            T_cells=T_cells.astype(np.float32),
            T_amb=T_amb,
            R_int=R_int.astype(np.float32),
            risk=risk.astype(np.float32),
            P_dendrite=P_dendrite,
            L_sei=L_sei,
            failure_step=failure_step,
            edge_index=graph["edge_index"].numpy(),
            edge_attr=graph["edge_attr"].numpy(),
            edge_type=graph["edge_type"].numpy(),
            capacity_multiplier=graph["capacity_multiplier"].numpy(),
            failure_cell=int(graph["failure_cell"].item()),
            failure_mode=int(graph["failure_mode"].item()),
        )
        np.savez_compressed(
            compat_path,
            V=V_cells.astype(np.float32),
            T=T_cells.astype(np.float32),
            I=np.repeat(I_pack[:, None], n_cells, axis=1).astype(np.float32),
            risk=risk.astype(np.float32),
            fail_cell=int(graph["failure_cell"].item()),
            failure_type=str(int(graph["failure_mode"].item())),
            failure_step=failure_step,
        )
        return {"path": str(path), "compat_path": str(compat_path), "n_cells": n_cells, "inject_failure": bool(inject_failure)}

    def run_batch(self, n_instances=50, duration_steps=900, family="mixed"):
        rows = []
        for i in range(n_instances):
            rows.append(self.generate_instance(i, duration_steps=duration_steps, family=family, inject_failure=i < max(1, n_instances // 3)))
        manifest = {"instances": len(rows), "rows": rows}
        (self.out_dir / "bms_pack_graph_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/synthetic/bms")
    parser.add_argument("--instances", type=int, default=50)
    parser.add_argument("--duration-steps", type=int, default=900)
    parser.add_argument("--family", choices=["series", "parallel", "mixed"], default="mixed")
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()
    gen = SyntheticBMSDataGenerator(args.out_dir, seed=args.seed)
    print(json.dumps(gen.run_batch(args.instances, args.duration_steps, args.family), indent=2))


if __name__ == "__main__":
    main()
