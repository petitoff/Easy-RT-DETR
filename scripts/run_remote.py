from __future__ import annotations

import argparse
import base64
import os
import tarfile
import tempfile
from datetime import timedelta
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from pyrun_jupyter import JupyterRunner
from pyrun_jupyter.exceptions import KernelError

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:  # pragma: no cover - dependency is optional at runtime
    Minio = None
    S3Error = Exception


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = Path(__file__).with_name(".env")
DEFAULT_EXCLUDES = [
    ".git",
    ".venv",
    ".cache",
    "artifacts",
    "__pycache__",
    ".pytest_cache",
    "*.pyc",
]


PRESETS: dict[str, dict[str, object]] = {
    "train-pennfudan": {
        "script": "scripts/train_pennfudan_cpu.py",
        "artifacts": ["artifacts/*.pt"],
        "description": "Train Penn-Fudan model remotely.",
    },
    "train-synthetic": {
        "script": "scripts/train_synthetic_cpu.py",
        "artifacts": ["artifacts/*.pt"],
        "description": "Train synthetic rectangles model remotely.",
        "exclude": ["data"],
    },
    "eval-pennfudan": {
        "script": "scripts/eval_pennfudan_checkpoint.py",
        "artifacts": [],
        "description": "Run Penn-Fudan checkpoint evaluation remotely.",
    },
    "visualize-pennfudan": {
        "script": "scripts/visualize_pennfudan_predictions.py",
        "artifacts": ["artifacts/**/*.png"],
        "description": "Render prediction visualizations remotely.",
    },
    "train-voc-car": {
        "script": "scripts/train_voc_car.py",
        "artifacts": ["artifacts/*.pt"],
        "description": "Train Pascal VOC car-only model remotely.",
    },
    "eval-voc-car": {
        "script": "scripts/eval_voc_car_checkpoint.py",
        "artifacts": [],
        "description": "Run Pascal VOC car-only checkpoint evaluation remotely.",
    },
    "visualize-voc-car": {
        "script": "scripts/visualize_voc_car_predictions.py",
        "artifacts": ["artifacts/**/*.png"],
        "description": "Render Pascal VOC car-only visualizations remotely.",
    },
}
UPLOAD_OK_MARKER = "__PYRUN_UPLOAD_OK__"
DOWNLOAD_START_MARKER = "__PYRUN_DOWNLOAD_START__"
DOWNLOAD_END_MARKER = "__PYRUN_DOWNLOAD_END__"
SIZE_MARKER = "__PYRUN_FILE_SIZE__"


def load_local_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_PATH.exists():
        return values
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def resolve_env_value(env_values: dict[str, str], *keys: str, default: str | None = None) -> str | None:
    for key in keys:
        value = os.environ.get(key) or env_values.get(key)
        if value:
            return value
    return default


def infer_default_dataset_source(preset: str) -> Path | None:
    if preset == "train-pennfudan":
        candidates = [
            PROJECT_ROOT / "data" / "PennFudanPed.zip",
            PROJECT_ROOT / "data" / "PennFudanPed",
        ]
    elif preset == "train-voc-car":
        candidates = [
            PROJECT_ROOT / "data" / "VOCdevkit",
            PROJECT_ROOT / "data" / "VOCtrainval_06-Nov-2007.tar",
        ]
    else:
        return None
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def prepare_dataset_archive(dataset_source: Path) -> tuple[Path, Path | None]:
    if dataset_source.is_file():
        return dataset_source, None

    temp_handle = tempfile.NamedTemporaryFile(prefix=f"{dataset_source.name}_", suffix=".tar.gz", delete=False)
    temp_path = Path(temp_handle.name)
    temp_handle.close()
    with tarfile.open(temp_path, "w:gz") as archive:
        archive.add(dataset_source, arcname=dataset_source.name)
    return temp_path, temp_path


def infer_dataset_root_name(dataset_source: Path) -> str:
    if dataset_source.is_dir():
        return dataset_source.name
    suffixes = "".join(dataset_source.suffixes)
    if suffixes in {".tar.gz", ".tar.bz2", ".tar.xz"}:
        return dataset_source.name[: -len(suffixes)]
    if dataset_source.suffix:
        return dataset_source.stem
    return dataset_source.name


def build_minio_client(endpoint: str, access_key: str, secret_key: str) -> Minio:
    if Minio is None:
        raise RuntimeError("Package `minio` is required for MinIO staging.")
    parsed = urlparse(endpoint)
    host = parsed.netloc or parsed.path
    secure = parsed.scheme == "https"
    if not host:
        raise ValueError(f"Invalid MinIO endpoint: {endpoint}")
    return Minio(host, access_key=access_key, secret_key=secret_key, secure=secure)


def ensure_bucket_exists(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def stage_dataset_via_minio(
    dataset_source: Path,
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    object_name: str | None,
    expiry_hours: int,
) -> dict[str, str]:
    archive_path, temp_archive = prepare_dataset_archive(dataset_source)
    try:
        client = build_minio_client(endpoint, access_key, secret_key)
        ensure_bucket_exists(client, bucket)
        if object_name is None:
            object_name = f"datasets/{archive_path.name}"
        local_size = archive_path.stat().st_size
        should_upload = True
        try:
            stat = client.stat_object(bucket, object_name)
            should_upload = stat.size != local_size
        except S3Error:
            should_upload = True
        if should_upload:
            client.fput_object(bucket, object_name, str(archive_path))
        url = client.presigned_get_object(bucket, object_name, expires=timedelta(hours=expiry_hours))
    finally:
        if temp_archive is not None and temp_archive.exists():
            temp_archive.unlink()

    archive_name = archive_path.name
    dataset_root_name = infer_dataset_root_name(dataset_source)
    return {
        "url": url,
        "archive_name": archive_name,
        "extract_dir": "data",
        "expected_root": f"data/{dataset_root_name}",
        "object_name": object_name,
        "bucket": bucket,
    }


def prepare_remote_artifact_via_minio(
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    object_name: str,
    expiry_hours: int,
) -> dict[str, str]:
    client = build_minio_client(endpoint, access_key, secret_key)
    ensure_bucket_exists(client, bucket)
    return {
        "put_url": client.presigned_put_object(bucket, object_name, expires=timedelta(hours=expiry_hours)),
        "get_url": client.presigned_get_object(bucket, object_name, expires=timedelta(hours=expiry_hours)),
        "bucket": bucket,
        "object_name": object_name,
    }


def download_minio_object(
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    object_name: str,
    destination: Path,
) -> Path:
    client = build_minio_client(endpoint, access_key, secret_key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.fget_object(bucket, object_name, str(destination))
    return destination


def extract_output_path(script_args: list[str]) -> str | None:
    for index, arg in enumerate(script_args):
        if arg == "--output" and index + 1 < len(script_args):
            return script_args[index + 1]
        if arg.startswith("--output="):
            return arg.split("=", 1)[1]
    return None


class ChunkedJupyterRunner(JupyterRunner):
    def __init__(self, *args, upload_chunk_bytes: int = 128 * 1024, download_chunk_bytes: int = 128 * 1024, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upload_chunk_bytes = upload_chunk_bytes
        self.download_chunk_bytes = download_chunk_bytes

    def _parse_marked_payload(self, stdout: str, start_marker: str, end_marker: str | None = None) -> str:
        if start_marker not in stdout:
            raise KernelError(f"Missing marker {start_marker!r} in remote response")
        payload = stdout.split(start_marker, 1)[1]
        if end_marker is not None:
            if end_marker not in payload:
                raise KernelError(f"Missing marker {end_marker!r} in remote response")
            payload = payload.split(end_marker, 1)[0]
        return payload.strip()

    def upload_via_kernel_chunked(self, local_path: Path, remote_path: str) -> bool:
        local_path = Path(local_path)
        init_result = self.run(f"""
import os

remote_path = {remote_path!r}
project_root = globals().get("_PYRUN_JUPYTER_ROOT")
if project_root is None:
    project_root = os.getcwd()
    globals()["_PYRUN_JUPYTER_ROOT"] = project_root
if not os.path.isabs(remote_path):
    remote_path = os.path.abspath(os.path.join(project_root, remote_path))
remote_dir = os.path.dirname(remote_path)
if remote_dir:
    os.makedirs(remote_dir, exist_ok=True)
with open(remote_path, "wb"):
    pass
print({UPLOAD_OK_MARKER!r})
""")
        if UPLOAD_OK_MARKER not in init_result.stdout:
            raise KernelError(f"Failed to initialize remote file: {remote_path}")

        with local_path.open("rb") as handle:
            while True:
                chunk = handle.read(self.upload_chunk_bytes)
                if not chunk:
                    break
                chunk_b64 = base64.b64encode(chunk).decode("ascii")
                result = self.run(f"""
import base64
import os

remote_path = {remote_path!r}
project_root = globals().get("_PYRUN_JUPYTER_ROOT")
if project_root is None:
    project_root = os.getcwd()
    globals()["_PYRUN_JUPYTER_ROOT"] = project_root
if not os.path.isabs(remote_path):
    remote_path = os.path.abspath(os.path.join(project_root, remote_path))
with open(remote_path, "ab") as handle:
    handle.write(base64.b64decode({chunk_b64!r}))
print({UPLOAD_OK_MARKER!r})
""", timeout=120.0)
                if UPLOAD_OK_MARKER not in result.stdout:
                    raise KernelError(f"Failed to upload chunk for {remote_path}")
        return True

    def _sync_project_chunked(
        self,
        project_dir: Path,
        remote_dir: str,
        exclude_patterns: list[str] | None = None,
    ) -> list[str]:
        files = self._iter_local_files(project_dir, pattern="**/*", exclude_patterns=exclude_patterns)
        files = sorted(files, key=lambda item: item[0].stat().st_size)
        uploaded: list[str] = []
        for local_path, rel_remote in files:
            remote_path = self._join_remote_path(remote_dir, rel_remote.as_posix())
            self.upload_via_kernel_chunked(local_path, remote_path)
            uploaded.append(remote_path)
        return uploaded

    def _remote_file_size(self, remote_path: str, working_dir: str = "") -> int:
        full_path = f"{working_dir}/{remote_path}" if working_dir else remote_path
        result = self.run(f"""
import os

filepath = {full_path!r}
project_root = globals().get("_PYRUN_JUPYTER_ROOT")
if project_root is None:
    project_root = os.getcwd()
    globals()["_PYRUN_JUPYTER_ROOT"] = project_root
if not os.path.isabs(filepath):
    filepath = os.path.abspath(os.path.join(project_root, filepath))
print({SIZE_MARKER!r})
print(os.path.getsize(filepath) if os.path.exists(filepath) else -1)
""", timeout=60.0)
        return int(self._parse_marked_payload(result.stdout, SIZE_MARKER))

    def download_kernel_files_chunked(
        self,
        remote_paths: list[str],
        local_dir: Path,
        working_dir: str = "",
        flatten: bool = False,
    ) -> list[Path]:
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        for remote_path in remote_paths:
            file_size = self._remote_file_size(remote_path, working_dir=working_dir)
            if file_size < 0:
                continue
            local_path = local_dir / Path(remote_path).name if flatten else local_dir / Path(remote_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with local_path.open("wb") as handle:
                for offset in range(0, file_size, self.download_chunk_bytes):
                    full_path = f"{working_dir}/{remote_path}" if working_dir else remote_path
                    result = self.run(f"""
import base64
import os

filepath = {full_path!r}
offset = {offset}
chunk_size = {self.download_chunk_bytes}
project_root = globals().get("_PYRUN_JUPYTER_ROOT")
if project_root is None:
    project_root = os.getcwd()
    globals()["_PYRUN_JUPYTER_ROOT"] = project_root
if not os.path.isabs(filepath):
    filepath = os.path.abspath(os.path.join(project_root, filepath))
with open(filepath, "rb") as source:
    source.seek(offset)
    data = source.read(chunk_size)
print({DOWNLOAD_START_MARKER!r})
print(base64.b64encode(data).decode("ascii"))
print({DOWNLOAD_END_MARKER!r})
""", timeout=120.0)
                    payload = self._parse_marked_payload(result.stdout, DOWNLOAD_START_MARKER, DOWNLOAD_END_MARKER)
                    handle.write(base64.b64decode(payload))
            downloaded.append(local_path)
        return downloaded

    def run_project_chunked(
        self,
        project_dir: Path,
        entrypoint: str,
        artifact_paths: list[str] | None,
        local_artifact_dir: Path,
        remote_dir: str,
        exclude_patterns: list[str] | None,
        params: dict[str, object],
        timeout: float,
    ):
        relative_entrypoint = self._normalize_entrypoint(project_dir, entrypoint)
        self._prepare_remote_project_dir(remote_dir)
        self._sync_project_chunked(project_dir, remote_dir, exclude_patterns=exclude_patterns)
        result = self.run(
            self._build_project_run_code(remote_dir, PurePosixPath(relative_entrypoint), params=params),
            timeout=timeout,
        )
        resolved_artifacts: list[str] = []
        if artifact_paths:
            resolved_artifacts = self._resolve_kernel_artifacts(artifact_paths, working_dir=remote_dir)
        local_paths: list[Path] = []
        if resolved_artifacts:
            local_paths = self.download_kernel_files_chunked(
                resolved_artifacts,
                local_dir=local_artifact_dir,
                working_dir=remote_dir,
                flatten=False,
            )
        result.data.setdefault("artifacts", [str(path) for path in local_paths])
        return result


def parse_args() -> argparse.Namespace:
    env_values = load_local_env()
    parser = argparse.ArgumentParser(description="Run Easy-RT-DETR scripts on a remote Jupyter server.")
    parser.add_argument("preset", choices=[*PRESETS.keys(), "custom"])
    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("PYRUN_JUPYTER_URL") or env_values.get("PYRUN_JUPYTER_URL") or env_values.get("KAGGLE_PROXY_URL"),
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("PYRUN_JUPYTER_TOKEN") or env_values.get("PYRUN_JUPYTER_TOKEN") or env_values.get("KAGGLE_PROXY_TOKEN"),
    )
    parser.add_argument("--kernel-name", type=str, default="python3")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--remote-dir", type=str, default="easy_rtdetr_remote")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/remote")
    parser.add_argument("--artifact", action="append", default=None, help="Extra artifact glob to download.")
    parser.add_argument("--exclude", action="append", default=None, help="Extra project sync exclude pattern.")
    parser.add_argument("--script", type=str, default=None, help="Script path relative to project root for custom preset.")
    parser.add_argument("--device", type=str, default=None, help="Override device passed to training script, e.g. cuda.")
    parser.add_argument("--upload-chunk-kb", type=int, default=128)
    parser.add_argument("--download-chunk-kb", type=int, default=128)
    parser.add_argument("--dataset-source", type=str, default=None, help="Local dataset file or directory to stage via MinIO.")
    parser.add_argument(
        "--minio-endpoint",
        type=str,
        default=resolve_env_value(env_values, "MINIO_ENDPOINT", "MINIO_URL"),
        help="MinIO S3 endpoint, e.g. http://host:9000.",
    )
    parser.add_argument(
        "--minio-access-key",
        type=str,
        default=resolve_env_value(env_values, "MINIO_ACCESS_KEY", "MINIO_ROOT_USER"),
    )
    parser.add_argument(
        "--minio-secret-key",
        type=str,
        default=resolve_env_value(env_values, "MINIO_SECRET_KEY", "MINIO_ROOT_PASSWORD"),
    )
    parser.add_argument(
        "--minio-bucket",
        type=str,
        default=resolve_env_value(env_values, "MINIO_BUCKET", default="dataset"),
    )
    parser.add_argument(
        "--dataset-object-name",
        type=str,
        default=None,
        help="Object name for the staged dataset archive inside the MinIO bucket.",
    )
    parser.add_argument("--minio-expiry-hours", type=int, default=24)
    args, script_args = parser.parse_known_args()
    args.script_args = script_args
    return args


def build_run_configuration(args: argparse.Namespace) -> tuple[str, list[str], list[str], list[str]]:
    if args.preset == "custom":
        if not args.script:
            raise ValueError("--script is required when preset=custom")
        script_path = args.script
        artifact_paths = []
        preset_excludes: list[str] = []
    else:
        preset = PRESETS[args.preset]
        script_path = str(preset["script"])
        artifact_paths = list(preset["artifacts"])
        preset_excludes = list(preset.get("exclude", []))

    script_args = list(args.script_args)
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    if args.device is not None and "--device" not in script_args and script_path.endswith(("train_pennfudan_cpu.py", "train_synthetic_cpu.py")):
        script_args = ["--device", args.device, *script_args]

    if args.artifact:
        artifact_paths.extend(args.artifact)
    return script_path, script_args, artifact_paths, preset_excludes


def main() -> None:
    args = parse_args()
    if not args.url:
        raise ValueError("Missing --url or PYRUN_JUPYTER_URL")

    script_path, script_args, artifact_paths, preset_excludes = build_run_configuration(args)
    exclude_patterns = [*DEFAULT_EXCLUDES, *preset_excludes, *(args.exclude or [])]
    local_artifact_dir = Path(args.artifact_dir) / args.preset.replace("/", "_")
    dataset_source_arg = Path(args.dataset_source) if args.dataset_source else infer_default_dataset_source(args.preset)
    remote_dataset: dict[str, str] | None = None
    remote_artifact: dict[str, str] | None = None
    minio_values = [args.minio_endpoint, args.minio_access_key, args.minio_secret_key]

    if dataset_source_arg is not None:
        dataset_source = dataset_source_arg.expanduser().resolve()
        if not dataset_source.exists():
            raise FileNotFoundError(f"Dataset source does not exist: {dataset_source}")
        if not all(minio_values):
            raise ValueError(
                "Dataset staging via MinIO requires --minio-endpoint, --minio-access-key and --minio-secret-key "
                "(or corresponding values in scripts/.env)."
            )
        remote_dataset = stage_dataset_via_minio(
            dataset_source=dataset_source,
            endpoint=args.minio_endpoint,
            access_key=args.minio_access_key,
            secret_key=args.minio_secret_key,
            bucket=args.minio_bucket,
            object_name=args.dataset_object_name,
            expiry_hours=args.minio_expiry_hours,
        )
        exclude_patterns.extend(["data", "data/**"])
        print(f"minio_bucket={remote_dataset['bucket']}")
        print(f"minio_object={remote_dataset['object_name']}")
        print(f"remote_dataset_expected_root={remote_dataset['expected_root']}")

    output_path = extract_output_path(script_args)
    if output_path is not None and all(minio_values):
        artifact_object_name = f"artifacts/{args.preset.replace('/', '_')}/{Path(output_path).name}"
        remote_artifact = prepare_remote_artifact_via_minio(
            endpoint=args.minio_endpoint,
            access_key=args.minio_access_key,
            secret_key=args.minio_secret_key,
            bucket=args.minio_bucket,
            object_name=artifact_object_name,
            expiry_hours=args.minio_expiry_hours,
        )
        remote_artifact["remote_path"] = output_path
        remote_artifact["local_path"] = str(local_artifact_dir / output_path)
        artifact_paths = []
        print(f"artifact_minio_bucket={remote_artifact['bucket']}")
        print(f"artifact_minio_object={remote_artifact['object_name']}")

    print(f"remote_url={args.url}")
    print(f"preset={args.preset}")
    print(f"script={script_path}")
    print(f"script_args={script_args}")
    print(f"artifact_paths={artifact_paths}")
    print(f"artifact_dir={local_artifact_dir}")

    with ChunkedJupyterRunner(
        args.url,
        token=args.token,
        kernel_name=args.kernel_name,
        upload_chunk_bytes=args.upload_chunk_kb * 1024,
        download_chunk_bytes=args.download_chunk_kb * 1024,
    ) as runner:
        result = runner.run_project_chunked(
            project_dir=PROJECT_ROOT,
            entrypoint="scripts/remote_dispatch.py",
            artifact_paths=artifact_paths or None,
            local_artifact_dir=local_artifact_dir,
            remote_dir=args.remote_dir,
            exclude_patterns=exclude_patterns,
            params={
                "remote_script": script_path,
                "remote_argv": script_args,
                "remote_dataset": remote_dataset,
                "remote_artifact": remote_artifact,
            },
            timeout=args.timeout,
        )

    if remote_artifact is not None:
        downloaded = download_minio_object(
            endpoint=args.minio_endpoint,
            access_key=args.minio_access_key,
            secret_key=args.minio_secret_key,
            bucket=remote_artifact["bucket"],
            object_name=remote_artifact["object_name"],
            destination=Path(remote_artifact["local_path"]),
        )
        result.data.setdefault("artifacts", [])
        result.data["artifacts"].append(str(downloaded))

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.data.get("artifacts"):
        print("downloaded_artifacts=")
        for path in result.data["artifacts"]:
            print(path)
    if not result.success:
        raise SystemExit(f"Remote execution failed: {result.error_name}: {result.error}")


if __name__ == "__main__":
    main()
