#!/usr/bin/env python3
"""Dependency-free release checks; never print matched credential values."""
import ast
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {'.git', '.cache', '__pycache__', '.venv', 'venv', 'build', 'dist'}
BINARY_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors', '.glb', '.ply', '.usdc', '.usd', '.usda', '.usdz', '.npz', '.npy', '.so', '.o', '.pem', '.key'}
SECRET_PATTERNS = [
    re.compile(r'ms-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}'),
    re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,})\b'),
    re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b'),
    re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
]


def paths():
    # Inspect Git's actual publication set, including untracked nonignored files.
    result = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                            cwd=ROOT, capture_output=True, check=True)
    for name in sorted(set(result.stdout.decode().split('\0')) - {''}):
        path = ROOT / name
        if path.is_file():
            yield path


def main():
    failures = []
    files = list(paths())
    python_count = shell_count = 0
    for path in files:
        rel = path.relative_to(ROOT)
        if path.is_symlink():
            failures.append(f'{rel}: symlink is not part of the source release')
            continue
        if any(p in IGNORED_PARTS for p in rel.parts) or rel.parts[0] in {'data', 'results', 'models', 'checkpoints', 'logs'}:
            failures.append(f'{rel}: runtime/generated path in publication set')
        if path.suffix in BINARY_SUFFIXES or (path.name.startswith('.env') and path.name != '.env.example'):
            failures.append(f'{rel}: credential/model/runtime file in publication set')
        if path.stat().st_size > 5_000_000:
            failures.append(f'{rel}: exceeds 5 MB source-file limit')
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            failures.append(f'{rel}: unexpected non-text file')
            continue
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            failures.append(f'{rel}: potential credential detected (value suppressed)')
        if re.search(r'/(?:home|Users)/[A-Za-z0-9_.-]+/(?:Project|anaconda|miniconda)', content):
            failures.append(f'{rel}: original machine-specific absolute path')
        if path.suffix == '.py':
            try:
                ast.parse(content, filename=str(rel))
                python_count += 1
            except SyntaxError as exc:
                failures.append(f'{rel}: Python syntax error at line {exc.lineno}')
        elif path.suffix == '.sh':
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            shell_count += 1
            if result.returncode:
                failures.append(f'{rel}: shell syntax check failed')
        elif path.suffix == '.md':
            for target in re.findall(r'\]\(([^\s)]+)\)', content):
                if re.match(r'(?:[a-z][a-z0-9+.-]*:|#)', target, re.I):
                    continue
                target = unquote(target.split('#', 1)[0])
                if target and not (path.parent / target).exists():
                    failures.append(f'{rel}: missing internal link {target}')
    for name in ['README.md', 'LICENSE', 'NOTICE', 'THIRD_PARTY.md', '.env.example',
                 'scripts/setup_scene_gen.sh', 'examples/mesh/assemble_full_object_scene.py',
                 'examples/segmentation/generate_worldsculpt_objects.py',
                 'submodules/CropFormer/mask2former/data/__init__.py']:
        if not (ROOT / name).is_file():
            failures.append(f'Missing required source: {name}')
    for message in failures:
        print(message, file=sys.stderr)
    print(f'Checked {len(files)} text source/config files, {python_count} Python files, {shell_count} shell files; failures={len(failures)}')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
