"""Sequential pairing fallback for upstream HLOC (which has no such module)."""
from collections import defaultdict
from pathlib import Path


def main(output, image_list, window_size=15, quadratic_overlap=True,
         use_loop_closure=True, retrieval_path=None, retrieval_interval=2,
         num_loc=3, groupby_folder=False):
    names = sorted(image_list)
    groups = defaultdict(list)
    for name in names:
        groups[str(Path(name).parent) if groupby_folder else ""].append(name)
    pairs = set()
    for group in groups.values():
        offsets = set(range(1, window_size + 1))
        if quadratic_overlap:
            offsets.update(2 ** k for k in range(1, len(group).bit_length()))
        for i, name in enumerate(group):
            for offset in offsets:
                if i + offset < len(group):
                    pairs.add(tuple(sorted((name, group[i + offset]))))
    if use_loop_closure and len(names) > 1:
        if retrieval_path is None:
            raise ValueError("Loop closure requires retrieval descriptors")
        from hloc import pairs_from_retrieval
        loop_path = Path(output).with_suffix('.loop.txt')
        pairs_from_retrieval.main(
            retrieval_path, loop_path, num_matched=min(num_loc, len(names) - 1),
            query_list=names[::max(1, retrieval_interval)], db_list=names,
        )
        for line in loop_path.read_text().splitlines():
            a, b = line.split()
            if a != b:
                pairs.add(tuple(sorted((a, b))))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(f'{a} {b}\n' for a, b in sorted(pairs)))
