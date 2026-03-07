"""CacheManager: checksum computation, manifest I/O, and cache validation.

Implements the Cacheable protocol. Uses SHA-256 of the first 1 MB plus
file size for fast checksumming of large PGN files.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

from chess_sim.data.streaming_types import ManifestInfo, PreprocessConfig

logger = logging.getLogger(__name__)

# Read first 1 MB for fast checksumming of large files.
_CHECKSUM_BYTES: int = 1_048_576


class CacheManager:
    """Manages preprocessing cache via checksums and manifest JSON files.

    Implements the Cacheable protocol. The manifest.json schema:
    {"source_checksum": str, "shard_paths": list[str],
     "total_examples": int, "examples_per_shard": list[int],
     "config": dict}.

    Example:
        >>> cm = CacheManager()
        >>> checksum = cm.checksum(Path("games.pgn"))
        >>> cm.is_cached(Path("output/manifest.json"), checksum)
        False
    """

    def checksum(self, path: Path) -> str:
        """Compute a fast checksum of a file.

        SHA-256 of first 1 MB concatenated with the file size.

        Args:
            path: Path to the file to checksum.

        Returns:
            Hex-encoded SHA-256 digest string.

        Example:
            >>> cm = CacheManager()
            >>> cs = cm.checksum(Path("data.pgn"))
            >>> len(cs)  # SHA-256 hex digest length
            64
        """
        h = hashlib.sha256()
        file_size = path.stat().st_size
        with open(path, "rb") as f:
            data = f.read(_CHECKSUM_BYTES)
            h.update(data)
        h.update(str(file_size).encode("ascii"))
        return h.hexdigest()

    def is_cached(
        self, manifest_path: Path, source_checksum: str
    ) -> bool:
        """Check whether a valid cached manifest exists.

        Returns False if the manifest file does not exist or if the
        stored checksum does not match the provided source_checksum.

        Args:
            manifest_path: Path to the manifest.json file.
            source_checksum: Expected checksum of the source PGN file.

        Returns:
            True if manifest exists and checksums match.

        Example:
            >>> cm.is_cached(Path("output/manifest.json"), "abc123")
            True
        """
        if not manifest_path.exists():
            return False
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("source_checksum") == source_checksum
        except (json.JSONDecodeError, OSError):
            logger.warning(
                "Failed to read manifest at %s", manifest_path
            )
            return False

    def write_manifest(
        self, info: ManifestInfo, output_dir: Path
    ) -> Path:
        """Serialize a ManifestInfo to manifest.json in output_dir.

        Creates the output directory if it does not exist.

        Args:
            info: ManifestInfo dataclass with all preprocessing
                metadata.
            output_dir: Directory where manifest.json will be written.

        Returns:
            Path to the written manifest.json file.

        Example:
            >>> path = cm.write_manifest(info, Path("/tmp/output"))
            >>> path.name
            'manifest.json'
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"

        data = {
            "source_checksum": info.source_checksum,
            "shard_paths": [str(p) for p in info.shard_paths],
            "total_examples": info.total_examples,
            "examples_per_shard": info.examples_per_shard,
            "config": asdict(info.config),
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(
            "Wrote manifest with %d shards, %d total examples",
            len(info.shard_paths),
            info.total_examples,
        )
        return manifest_path

    def load_manifest(self, path: Path) -> ManifestInfo:
        """Load a ManifestInfo from a manifest.json file on disk.

        Args:
            path: Path to the manifest.json file.

        Returns:
            ManifestInfo dataclass reconstructed from JSON.

        Raises:
            FileNotFoundError: If the manifest file does not exist.

        Example:
            >>> info = cm.load_manifest(Path("output/manifest.json"))
            >>> info.total_examples
            42000
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = PreprocessConfig(**data["config"])
        return ManifestInfo(
            source_checksum=data["source_checksum"],
            shard_paths=[Path(p) for p in data["shard_paths"]],
            total_examples=data["total_examples"],
            examples_per_shard=data["examples_per_shard"],
            config=config,
        )
