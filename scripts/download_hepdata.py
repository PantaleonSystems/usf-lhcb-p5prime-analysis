from __future__ import annotations

import io
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
ARCHIVE_NAME = "HEPData-ins1409497-v1-yaml"
DOWNLOAD_URL = "https://www.hepdata.net/download/submission/ins1409497/1/yaml"
STAMP = RAW_DIR / ARCHIVE_NAME / ".ins1409497-v1-downloaded"


def download_hepdata() -> None:
    destination = RAW_DIR / ARCHIVE_NAME
    required = (destination / "Table1.yaml", destination / "Table2.yaml")
    if all(path.is_file() and path.stat().st_size for path in required):
        STAMP.parent.mkdir(parents=True, exist_ok=True)
        STAMP.touch()
        print(f"Using cached HEPData YAML in {destination.relative_to(REPOSITORY_ROOT)}")
        return

    try:
        response = subprocess.run(
            [
                "curl", "--fail", "--silent", "--show-error", "--location",
                "--retry", "3", "--retry-all-errors", "--max-time", "120",
                DOWNLOAD_URL,
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("curl is required to download the HEPData archive") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Could not download HEPData archive: {detail}") from exc
    archive_bytes = response.stdout

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hepdata-") as temporary_directory:
        staging = Path(temporary_directory) / ARCHIVE_NAME
        staging.mkdir(parents=True)
        try:
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
                for member in archive.getmembers():
                    member_path = PurePosixPath(member.name)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise RuntimeError(f"Unsafe path in HEPData archive: {member.name}")
                    if not member_path.parts or member_path.parts[0] != ARCHIVE_NAME:
                        raise RuntimeError(f"Unexpected path in HEPData archive: {member.name}")
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise RuntimeError(f"Unsupported file in HEPData archive: {member.name}")

                    relative_path = Path(*member_path.parts[1:])
                    if not relative_path.parts:
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"Could not read archive member: {member.name}")
                    target = staging / relative_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
        except (tarfile.TarError, OSError) as exc:
            raise RuntimeError("Could not unpack the HEPData YAML archive") from exc

        if not all((staging / path.name).is_file() for path in required):
            raise RuntimeError("HEPData archive is missing required Table1.yaml or Table2.yaml")

        destination.mkdir(parents=True, exist_ok=True)
        for source in staging.rglob("*"):
            if source.is_file():
                target = destination / source.relative_to(staging)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)

    if not all(path.is_file() and path.stat().st_size for path in required):
        raise RuntimeError(f"HEPData download did not produce required files in {destination}")

    STAMP.touch()
    print(f"Downloaded HEPData ins1409497 v1 to {destination.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    download_hepdata()