#!/usr/bin/env python3
"""Download the pinned geometry-only WorldSculpt weights (no texture or 512 stage)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import requests

REPOS = {
    'TencentARC/Pixal3D': ('b0cb2e1b794cab9aa0ac38a95d794a4d9337437f', 'Pixal3D'),
    'AlayaLab/WorldSculpt': ('8cb81056d803c61371dd84ef18a14142a738610e', ''),
    'camenduru/dinov3-vitl16-pretrain-lvd1689m': ('3c276edd87d6f6e569ff0c4400e086807d0f3881', 'dinov3'),
}
BASES = ('shape_dec_next_dc_f16c32_fp16', 'ss_dec_conv3d_16l8_fp16',
         'ss_flow_img_dit_1_3B_64_bf16', 'slat_flow_img2shape_dit_1_3B_1024_bf16')

def selected(repo, name):
    if repo == 'TencentARC/Pixal3D':
        return name == 'pipeline.json' or name in {
            f'ckpts/{base}.{suffix}' for base in BASES for suffix in ('json', 'safetensors')}
    if repo == 'AlayaLab/WorldSculpt':
        return name.endswith(('.json', '.pt'))
    return name in ('model.safetensors', 'config.json', 'preprocessor_config.json', 'LICENSE.md')

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def download(entry, root):
    path = root / entry['local_path']
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = entry.get('sha256')
    if path.exists() and path.stat().st_size == entry['size']:
        if not expected or digest(path) == expected:
            print('verified', entry['local_path'], flush=True)
            return
    partial = path.with_name(path.name + '.partial')
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with requests.get(entry['url'], headers=headers, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                append = bool(offset and response.status_code == 206)
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Incorrect resume response')
                with partial.open('ab' if append else 'wb') as f:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        f.write(chunk)
            if partial.stat().st_size != entry['size']:
                raise RuntimeError('Download size mismatch')
            if expected and digest(partial) != expected:
                partial.unlink()
                raise RuntimeError('SHA256 mismatch')
            partial.replace(path)
            print('downloaded', entry['local_path'], entry['size'], flush=True)
            return
        except (requests.RequestException, RuntimeError) as exc:
            # Report type only: CDN signed URLs should not enter permanent logs.
            print('retry', entry['local_path'], type(exc).__name__, attempt + 1, flush=True)
            if attempt == 4:
                raise RuntimeError(f"Download failed: {entry['local_path']}") from None
            time.sleep(2 ** attempt)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.cache/scene_gen/models/worldsculpt'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    entries = []
    for repo, (revision, subdir) in REPOS.items():
        response = requests.get(f'https://huggingface.co/api/models/{repo}/revision/{revision}',
                                params={'blobs': 'true'}, timeout=60)
        response.raise_for_status()
        for item in response.json()['siblings']:
            name = item['rfilename']
            if not selected(repo, name):
                continue
            entries.append(dict(repo=repo, revision=revision, file=name, size=item['size'],
                                local_path=str(Path(subdir) / name),
                                sha256=item.get('lfs', {}).get('sha256'),
                                url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}'))
    need = sum(max(0, e['size'] - ((args.output/e['local_path']).stat().st_size
                       if (args.output/e['local_path']).exists() else 0)) for e in entries)
    if shutil.disk_usage(args.output).free < need + 4 * 1024 ** 3:
        raise SystemExit('Not enough disk space: reserve another 4 GiB for compilation/inference.')
    (args.output/'download_manifest.json').write_text(json.dumps(entries, indent=2) + '\n')
    print(f'{len(entries)} files; {need / 1e9:.2f} GB remaining', flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda e: download(e, args.output), entries))
    print('All geometry weights verified.', flush=True)

if __name__ == '__main__':
    main()
