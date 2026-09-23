#!/usr/bin/env python3
"""Download the SAM checkpoints from ModelScope with resume and SHA256 checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

import requests

ROOT = Path(__file__).resolve().parents[1]
REPOS = {"sam3": "facebook/sam3", "sam3d": "facebook/sam-3d-objects"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(repo, entry, output):
    relative = Path(entry["Path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid repository path: {relative}")
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = entry["Sha256"]
    if target.is_file() and target.stat().st_size == entry["Size"] and sha256(target) == expected:
        print(f"[verified] {relative}", flush=True)
        return
    if target.exists():
        raise RuntimeError(f"Existing file failed verification: {target}; inspect it before retrying.")
    temporary = target.with_name(target.name + ".part")
    url = requests.Request("GET", f"https://modelscope.cn/api/v1/models/{repo}/repo",
                           params={"Revision": "master", "FilePath": entry["Path"]}).prepare().url
    print(f"[download] {relative} ({entry['Size'] / 1e9:.2f} GB)", flush=True)
    subprocess.run(["curl", "-fL", "--silent", "--show-error", "-C", "-",
                    "--retry", "5", "--retry-all-errors", "--connect-timeout", "30",
                    "--max-time", "3600", url, "-o", str(temporary)], check=True)
    if temporary.stat().st_size != entry["Size"] or sha256(temporary) != expected:
        raise RuntimeError(f"Downloaded file failed SHA256/size verification: {temporary}")
    temporary.replace(target)
    print(f"[verified] {relative}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=REPOS, default="sam3")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    repo = REPOS[args.model]
    output = args.output_dir or ROOT / ".cache/scene_gen/models" / repo.split("/")[1]
    response = requests.get(f"https://modelscope.cn/api/v1/models/{repo}/repo/files",
                            params={"Revision": "master", "Recursive": "true"}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("Success"):
        raise RuntimeError(payload.get("Message", "ModelScope request failed"))
    entries = []
    for entry in payload["Data"]["Files"]:
        path = entry["Path"]
        wanted = path in {"LICENSE", "README.md"}
        if args.model == "sam3":
            wanted |= path == "sam3.pt"
        else:
            wanted |= path.startswith("checkpoints/") and path.endswith((".ckpt", ".yaml"))
        if wanted and entry["Size"] > 0:
            entries.append(entry)
    if not entries or not any(e["Path"].endswith((".pt", ".ckpt")) for e in entries):
        raise RuntimeError("No checkpoint files found in ModelScope repository")
    output.mkdir(parents=True, exist_ok=True)
    print(f"{repo}: {len(entries)} files, {sum(e['Size'] for e in entries) / 1e9:.2f} GB", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(lambda e: download(repo, e, output), entries))
    (output / "download_manifest.json").write_text(
        json.dumps({"source": "ModelScope", "repo": repo, "revision": "master",
                    "files": [{k: e[k] for k in ("Path", "Size", "Sha256")} for e in entries]}, indent=2) + "\n"
    )
    print(f"Verified model files: {output}", flush=True)


if __name__ == "__main__":
    main()
