"""Model Registry — V4 Architecture: load by name and version, not file path.

Every registered model has:
    - architecture hash (sha256 of model class source)
    - training data hash (sha256 of manifest used during training)
    - validation metrics snapshot
    - checkpoint path
    - timestamp

This is a local JSON-backed registry. Production upgrade: replace backing
store with MLflow or a cloud model registry.
"""

import hashlib
import inspect
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


REGISTRY_FILENAME = "model_registry.json"


@dataclass
class ModelCard:
    name: str
    version: str
    checkpoint_path: str
    architecture_hash: str
    training_data_hash: str
    validation_metrics: Dict[str, float]
    physics_terms: List[str]
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    claim_level: str = "simulation-backed"
    notes: str = ""


class ModelRegistry:
    """JSON-backed model registry living alongside checkpoints."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.registry_path = self.root / "checkpoints" / REGISTRY_FILENAME
        self._cards: Dict[str, Dict[str, ModelCard]] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            try:
                raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
                for name, versions in raw.items():
                    self._cards[name] = {}
                    for ver, card_dict in versions.items():
                        self._cards[name][ver] = ModelCard(**card_dict)
            except Exception:
                self._cards = {}

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for name, versions in self._cards.items():
            serializable[name] = {ver: asdict(card) for ver, card in versions.items()}
        self.registry_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    def register(self, card: ModelCard) -> None:
        """Register a new model version."""
        if card.name not in self._cards:
            self._cards[card.name] = {}
        self._cards[card.name][card.version] = card
        self._save()

    def get(self, name: str, version: str = "latest") -> Optional[ModelCard]:
        """Retrieve a model card by name and version."""
        versions = self._cards.get(name, {})
        if not versions:
            return None
        if version == "latest":
            return max(versions.values(), key=lambda c: c.created_at)
        return versions.get(version)

    def list_models(self) -> Dict[str, List[str]]:
        """List all registered model names and their versions."""
        return {name: sorted(versions.keys()) for name, versions in self._cards.items()}

    def summary(self) -> List[Dict[str, Any]]:
        """Summary table suitable for API or dashboard display."""
        rows = []
        for name, versions in self._cards.items():
            for ver, card in versions.items():
                rows.append({
                    "name": card.name,
                    "version": card.version,
                    "claim_level": card.claim_level,
                    "checkpoint": card.checkpoint_path,
                    "created_at": card.created_at,
                    "metrics": card.validation_metrics,
                })
        return rows

    @staticmethod
    def architecture_hash(model_class: type) -> str:
        """SHA256 of the model class source code."""
        try:
            source = inspect.getsource(model_class)
            return hashlib.sha256(source.encode()).hexdigest()[:16]
        except (TypeError, OSError):
            return "source_unavailable"

    @staticmethod
    def data_hash(manifest_path: Path) -> str:
        """SHA256 of the training data manifest file."""
        if manifest_path.exists():
            return hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]
        return "manifest_missing"


def register_from_training(
    name: str,
    version: str,
    checkpoint_path: str,
    model_class: type,
    manifest_path: Path,
    validation_metrics: Dict[str, float],
    physics_terms: Optional[List[str]] = None,
    notes: str = "",
    registry_root: Optional[Path] = None,
) -> ModelCard:
    """Convenience function to register a model after a training run."""
    registry = ModelRegistry(registry_root)
    card = ModelCard(
        name=name,
        version=version,
        checkpoint_path=str(checkpoint_path),
        architecture_hash=ModelRegistry.architecture_hash(model_class),
        training_data_hash=ModelRegistry.data_hash(manifest_path),
        validation_metrics=validation_metrics,
        physics_terms=physics_terms or [],
        claim_level="simulation-backed",
        notes=notes,
    )
    registry.register(card)
    return card
