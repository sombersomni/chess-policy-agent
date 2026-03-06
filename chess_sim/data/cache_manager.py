"""CacheManager: checksum computation, manifest I/O, and cache validation.

Implements the Cacheable protocol. Uses SHA-256 of the first 1 MB plus
file size for fast checksumming of large PGN files.
"""

from __future__ import annotations

from pathlib import Path

from chess_sim.data.streaming_types import ManifestInfo


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
        """Compute a fast checksum of a file (SHA-256 of first 1MB + file size).

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
        raise NotImplementedError("To be implemented")

    def is_cached(self, manifest_path: Path, source_checksum: str) -> bool:
        """Check whether a valid cached manifest exists for the given checksum.

        Returns False if the manifest file does not exist or if the stored
        checksum does not match the provided source_checksum.

        Args:
            manifest_path: Path to the manifest.json file.
            source_checksum: Expected checksum of the source PGN file.

        Returns:
            True if manifest exists and checksums match, False otherwise.

        Example:
            >>> cm.is_cached(Path("output/manifest.json"), "abc123")
            True
        """
        raise NotImplementedError("To be implemented")

    def write_manifest(self, info: ManifestInfo, output_dir: Path) -> Path:
        """Serialize a ManifestInfo to manifest.json in output_dir.

        Creates the output directory if it does not exist.

        Args:
            info: ManifestInfo dataclass with all preprocessing metadata.
            output_dir: Directory where manifest.json will be written.

        Returns:
            Path to the written manifest.json file.

        Example:
            >>> path = cm.write_manifest(info, Path("/tmp/output"))
            >>> path.name
            'manifest.json'
        """
        raise NotImplementedError("To be implemented")

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
        raise NotImplementedError("To be implemented")
