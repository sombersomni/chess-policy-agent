#!/usr/bin/env python3
"""File watcher that runs ruff lint on file changes."""

import subprocess
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class RuffHandler(FileSystemEventHandler):
    def __init__(self, watch_paths: list[Path]) -> None:
        self.watch_paths = watch_paths

    def _run_ruff(self, path: Path, fix: bool = False) -> None:
        cmd = ["python", "-m", "ruff", "check", str(path)]
        if fix:
            cmd.append("--fix")
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result

    def on_modified(self, event: Path) -> None:
        if event.is_directory:
            return
        if event.src_path.endswith(".py"):
            print(f"Checking {event.src_path}...")
            result = self._run_ruff(Path(event.src_path), fix=False)
            if result.returncode != 0:
                print("  Lint errors found, running fix...")
                fix_result = self._run_ruff(Path(event.src_path), fix=True)
                if fix_result.returncode == 0:
                    print("  Fixed!")
                else:
                    print("  Errors after fix:")
                    print(fix_result.stdout)
                    print(fix_result.stderr)
            else:
                print("  OK")

    def on_created(self, event: Path) -> None:
        if event.is_directory:
            return
        if event.src_path.endswith(".py"):
            print(f"Checking new file {event.src_path}...")
            result = self._run_ruff(Path(event.src_path), fix=False)
            if result.returncode != 0:
                print("  Lint errors found, running fix...")
                fix_result = self._run_ruff(Path(event.src_path), fix=True)
                if fix_result.returncode == 0:
                    print("  Fixed!")
                else:
                    print("  Errors after fix:")
                    print(fix_result.stdout)
                    print(fix_result.stderr)


def main() -> None:
    watch_paths = [Path(".").resolve()]
    event_handler = RuffHandler(watch_paths)
    observer = Observer()

    for path in watch_paths:
        observer.schedule(event_handler, str(path), recursive=True)

    observer.start()
    print(f"Watching for .py file changes in {watch_paths}...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
