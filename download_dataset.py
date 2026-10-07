"""List and download the Makerere maize image archives from Harvard Dataverse.

The dataset is public, so no Dataverse API token is required.
"""
import argparse
import hashlib
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PERSISTENT_ID = "doi:10.7910/DVN/LPGHKK"
DEFAULT_SERVER = "https://dataverse.harvard.edu"
USER_AGENT = "Mozilla/5.0 (compatible; MaizeDatasetDownloader/1.0)"
CLASS_DIRS = {
    "healthy": "healthy",
    "blight": "maize_leaf_blight",
    "streak": "maize_streak_virus",
}
CHUNK_SIZE = 1024 * 1024


def classify_archive(filename):
    """Return the project class folder for a Makerere archive, or None."""
    name = filename.lower()
    if name.startswith("healthy") and name.endswith(".rar"):
        return CLASS_DIRS["healthy"]
    if name.startswith("mlb") and name.endswith(".rar"):
        return CLASS_DIRS["blight"]
    if name.startswith("msv") and name.endswith(".rar"):
        return CLASS_DIRS["streak"]
    return None


def get_dataset_files(server):
    query = urlencode({"persistentId": PERSISTENT_ID})
    url = (
        server.rstrip("/")
        + "/api/datasets/:persistentId/versions/:latest-published/files?"
        + query
    )
    try:
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not retrieve dataset metadata from {url}: {exc}") from exc

    if payload.get("status") != "OK":
        raise RuntimeError(f"Dataverse API returned an error: {payload.get('message', payload)}")

    archives = []
    for entry in payload.get("data", []):
        data_file = entry.get("dataFile") or {}
        filename = data_file.get("filename") or entry.get("label", "")
        class_dir = classify_archive(filename)
        if class_dir is None:
            continue
        file_id = data_file.get("id")
        size = data_file.get("filesize")
        checksum = data_file.get("md5")
        if not isinstance(file_id, int) or not isinstance(size, int) or not checksum:
            raise RuntimeError(f"Dataverse metadata is incomplete for {filename!r}.")
        archives.append({
            "id": file_id,
            "filename": filename.replace("\\", "/").rsplit("/", 1)[-1],
            "size": size,
            "md5": checksum.lower(),
            "class_dir": class_dir,
        })

    if not archives:
        raise RuntimeError("No Healthy, MLB, or MSV RAR archives were listed by Dataverse.")
    return archives


def file_md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_size(size):
    return f"{size / (1024 ** 3):.2f} GiB"


def download_archive(server, archive, output_dir):
    destination_dir = os.path.join(output_dir, archive["class_dir"])
    os.makedirs(destination_dir, exist_ok=True)
    destination = os.path.join(destination_dir, archive["filename"])
    partial = destination + ".part"

    if os.path.exists(destination):
        if os.path.getsize(destination) == archive["size"] and file_md5(destination) == archive["md5"]:
            print(f"Already verified: {destination}")
            return
        print(f"Existing file is incomplete or has a checksum mismatch; replacing: {destination}")

    offset = os.path.getsize(partial) if os.path.exists(partial) else 0
    if offset > archive["size"]:
        os.remove(partial)
        offset = 0
    if offset == archive["size"]:
        if file_md5(partial) == archive["md5"]:
            os.replace(partial, destination)
            print(f"Verified: {destination}")
            return
        os.remove(partial)
        offset = 0

    url = f"{server.rstrip('/')}/api/access/datafile/{archive['id']}"
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=90) as response:
            status = response.getcode() or 200
            if status not in (200, 206):
                raise RuntimeError(f"Dataverse returned HTTP {status} for {archive['filename']}.")
            append = offset > 0 and status == 206
            if offset and not append:
                offset = 0
            mode = "ab" if append else "wb"
            downloaded = offset
            last_report = time.monotonic()
            with open(partial, mode) as file:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    file.write(chunk)
                    downloaded += len(chunk)
                    if time.monotonic() - last_report >= 3:
                        print(
                            f"\r{archive['filename']}: {downloaded / archive['size'] * 100:.1f}% "
                            f"({format_size(downloaded)} / {format_size(archive['size'])})",
                            end="",
                            flush=True,
                        )
                        last_report = time.monotonic()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Download failed for {archive['filename']}. Its partial file is kept so you can resume: {exc}"
        ) from exc

    actual_size = os.path.getsize(partial)
    if actual_size != archive["size"]:
        raise RuntimeError(
            f"Incomplete download for {archive['filename']}: expected {archive['size']} bytes, got {actual_size}. "
            "Run the same command again to resume."
        )
    if file_md5(partial) != archive["md5"]:
        raise RuntimeError(
            f"MD5 checksum mismatch for {archive['filename']}. The partial file is kept; "
            "remove it and retry if the mismatch persists."
        )

    os.replace(partial, destination)
    print(f"\rVerified and saved: {destination} ({format_size(archive['size'])})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Dataverse server URL.")
    parser.add_argument("--output", default=os.path.join("dataset", "raw"), help="Archive download folder.")
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=("all", *CLASS_DIRS),
        default=["all"],
        metavar="CLASS",
        help="Select all or one or more of: healthy, blight, streak (default: all).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download selected RAR archives. Without this flag, only list files and sizes.",
    )
    args = parser.parse_args()

    try:
        archives = get_dataset_files(args.server)
        selected = archives if "all" in args.classes else [
            archive for archive in archives
            if archive["class_dir"] in {CLASS_DIRS[name] for name in args.classes}
        ]
        if not selected:
            raise RuntimeError("No dataset archives matched the selected classes.")

        print("Makerere University Maize Image Dataset (Harvard Dataverse)")
        for archive in selected:
            print(
                f"  {archive['class_dir']:22} {archive['filename']:16} "
                f"{format_size(archive['size'])}  file id {archive['id']}"
            )
        total = sum(archive["size"] for archive in selected)
        print(f"Selected total: {format_size(total)}")
        if not args.download:
            print("Review the download size, then run again with --download to fetch these archives.")
            return

        for archive in selected:
            download_archive(args.server, archive, args.output)
        print(f"All selected archives are downloaded and checksum-verified under {args.output}.")
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
