from __future__ import annotations

import runpy
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as target:
        shutil.copyfileobj(response, target)


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)
        return
    with tarfile.open(archive_path) as archive:
        archive.extractall(extract_dir)


def bootstrap_dataset() -> None:
    if not remote_dataset:
        return

    extract_dir = Path(remote_dataset.get("extract_dir", "data"))
    expected_root = Path(remote_dataset.get("expected_root", extract_dir.as_posix()))
    if not extract_dir.is_absolute():
        extract_dir = Path.cwd() / extract_dir
    if not expected_root.is_absolute():
        expected_root = Path.cwd() / expected_root

    if expected_root.exists():
        print(f"remote_dataset_ready={expected_root}")
        return

    archive_name = remote_dataset.get("archive_name") or "dataset_archive"
    archive_path = Path.cwd() / ".cache" / archive_name
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"remote_dataset_download={remote_dataset['url']}")
    download_file(remote_dataset["url"], archive_path)
    extract_archive(archive_path, extract_dir)
    print(f"remote_dataset_ready={expected_root}")


def main() -> None:
    bootstrap_dataset()
    script = Path(remote_script)
    if not script.is_absolute():
        script = Path.cwd() / script
    argv = [str(arg) for arg in remote_argv]

    print(f"remote_dispatch_script={script}")
    print(f"remote_dispatch_argv={argv}")

    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
