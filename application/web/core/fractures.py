import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

MESHES_DIR = Path(__file__).resolve().parent / "static" / "core" / "meshes"


@dataclass
class Fracture:
    id: str
    category: str
    mesh_hash: str
    fracture: str
    num_pieces: int
    piece_type: str  # "two_pieces" / "multi_pieces"
    obj_paths: list[Path] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return f"{self.category} — {self.fracture} ({self.num_pieces} pieces)"

    @property
    def static_prefix(self) -> str:
        """Relative path usable with Django's {% static %} tag."""
        return f"core/meshes/{self.piece_type}/{self.category}/{self.mesh_hash}/{self.fracture}"


def _make_id(piece_type: str, category: str, mesh_hash: str, fracture: str) -> str:
    raw = f"{piece_type}/{category}/{mesh_hash}/{fracture}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def discover_fractures() -> list[Fracture]:
    """Walk the meshes directory and return all valid fractures."""
    fractures = []

    for piece_type in ("two_pieces", "multi_pieces"):
        type_dir = MESHES_DIR / piece_type
        if not type_dir.is_dir():
            continue

        for category_dir in sorted(type_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category = category_dir.name

            for hash_dir in sorted(category_dir.iterdir()):
                if not hash_dir.is_dir():
                    continue
                mesh_hash = hash_dir.name

                for fracture_dir in sorted(hash_dir.iterdir()):
                    if not fracture_dir.is_dir():
                        continue
                    if not fracture_dir.name.startswith("fractured_"):
                        continue

                    obj_files = sorted(fracture_dir.glob("piece_*.obj"))
                    if len(obj_files) < 2:
                        continue

                    fracture = Fracture(
                        id=_make_id(piece_type, category, mesh_hash, fracture_dir.name),
                        category=category,
                        mesh_hash=mesh_hash,
                        fracture=fracture_dir.name,
                        num_pieces=len(obj_files),
                        piece_type=piece_type,
                        obj_paths=obj_files,
                    )
                    fractures.append(fracture)

    return fractures


_cached_fractures: list[Fracture] | None = None
def get_fractures() -> list[Fracture]:
    global _cached_fractures
    if _cached_fractures is None:
        _cached_fractures = discover_fractures()
    return _cached_fractures


def get_fracture_by_id(fracture_id: str) -> Fracture | None:
    for f in get_fractures():
        if f.id == fracture_id:
            return f
    return None
