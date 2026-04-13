import hashlib
import json
from pathlib import Path

def calculate_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

class HashTracker:
    def __init__(self, brain_dir: Path, project_root: Path | None = None):
        self.hash_file = brain_dir / "hashes.json"
        self.project_root = project_root
        self.hashes = self._load()

    def _load(self) -> dict:
        if self.hash_file.exists():
            try:
                data = json.loads(self.hash_file.read_text())
                if any(Path(k).is_absolute() for k in data):
                    return {}
                return data
            except json.JSONDecodeError:
                return {}
        return {}

    def save(self):
        self.hash_file.write_text(json.dumps(self.hashes, indent=2))

    def has_changed(self, path: Path) -> bool:
        if not path.exists():
            return False
        if self.project_root is not None:
            try:
                rel_path = str(path.relative_to(self.project_root))
            except ValueError:
                rel_path = str(path)
        else:
            rel_path = str(path)
        current_hash = calculate_hash(path)
        if self.hashes.get(rel_path) == current_hash:
            return False
        self.hashes[rel_path] = current_hash
        return True

    def mark_stale(self, rel_path: str):
        """Force a file to be seen as changed."""
        if rel_path in self.hashes:
            del self.hashes[rel_path]
