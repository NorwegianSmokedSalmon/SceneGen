#!/usr/bin/env python3
"""Report fragmentation and input-view support without equating closure with completeness."""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]


def audit(root):
    choices = json.loads((root / 'selected_meshes.json').read_text())['objects']
    inventory = {o['instance_id']: o for o in json.loads((root / 'inventory.json').read_text())['objects']}
    rows = []
    for record in choices:
        obj = inventory[record['instance_id']]
        mesh = trimesh.load(root / record['repaired_mesh'], force='mesh', process=False)
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError(f"Invalid mesh: {record['repaired_mesh']}")
        components = trimesh.graph.connected_components(
            mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
        areas = sorted((float(mesh.area_faces[c].sum()) for c in components), reverse=True)
        meta = json.loads((root / record['source_mesh']).parent.parent.joinpath('transforms.json').read_text())
        cameras = np.asarray([np.asarray(f['source_camera']['c2w'])[:3, 3] for f in meta['frames']])
        directions = cameras - np.asarray(obj['offset'])
        distances = np.linalg.norm(directions, axis=1)
        if not len(cameras) or np.any(distances <= 1e-9):
            raise ValueError(f"Invalid camera directions for {obj['name']}")
        directions /= distances[:, None]
        angle = float(np.degrees(np.arccos(np.clip(directions @ directions.T, -1, 1))).max())
        row = {
            'name': obj['name'], 'instance_id': record['instance_id'],
            'selected_mesh': record['repaired_mesh'],
            'conditioning_views': len(cameras),
            'maximum_pairwise_camera_direction_angle_degrees': angle,
            'connected_components': len(components),
            'largest_component_area_fraction': areas[0] / mesh.area,
            'second_component_area_fraction': areas[1] / mesh.area if len(areas) > 1 else 0.,
            'watertight': bool(mesh.is_watertight),
            'winding_consistent': bool(mesh.is_winding_consistent),
            'source_candidate_observed_mask_recall': record.get('observed_mask_recall'),
            'source_candidate_silhouette_iou': record.get('silhouette_iou'),
            'repair_method': record.get('repair_method'),
        }
        completion = root / obj['directory'] / 'completed/completion.json'
        if completion.exists():
            data = json.loads(completion.read_text())
            row['api_inpainting_accepted_views'] = data['accepted_views']
            row['api_inpainting_attempted_views'] = len(data['views'])
        rows.append(row)
    return {
        'status': 'diagnostic_only',
        'semantic_completeness': 'not_certified',
        'scope': 'Read-only diagnostic of selected meshes. Fragment count is not a universal failure rule. Maximum pairwise camera angle is not a percentage of surface coverage. Source-candidate silhouette metrics precede repair and do not measure hidden surfaces.',
        'object_count': len(rows),
        'watertight_count': sum(r['watertight'] for r in rows),
        'source_candidates_with_observed_mask_recall_below_0_85': sum(
            r['source_candidate_observed_mask_recall'] is not None and r['source_candidate_observed_mask_recall'] < .85 for r in rows),
        'narrow_voxel_repaired_objects': sum((r['repair_method'] or '').startswith('Narrow voxel') for r in rows),
        'objects': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'results/scene_gen_room/full_objects')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    output = args.output or args.root / 'scene/geometry_completeness_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(f"Audited {result['object_count']} objects; {result['watertight_count']} watertight. Surface completeness is NOT certified.")
    print(output)


if __name__ == '__main__':
    main()
