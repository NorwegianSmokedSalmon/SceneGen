#!/usr/bin/env python3
"""Repair one generated mesh, preserving separate meaningful components."""
import argparse,json,time
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy import ndimage
from skimage.measure import marching_cubes
import open3d as o3d
import pymeshfix


def topology(mesh):
    edges=np.sort(np.concatenate([mesh.faces[:,[0,1]],mesh.faces[:,[1,2]],mesh.faces[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    return {'faces':len(mesh.faces),'vertices':len(mesh.vertices),'boundary_edges':int((counts==1).sum()),
            'nonmanifold_edges':int((counts>2).sum()),'watertight':bool((counts==2).all()),
            'bounds':mesh.bounds.tolist()}


def surface_distances(mesh,points):
    tensor_mesh=o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),o3d.utility.Vector3iVector(np.asarray(mesh.faces))))
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(tensor_mesh)
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points,np.float32))).numpy()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--voxel-only',action='store_true');p.add_argument('--preserve-cavities',action='store_true');p.add_argument('--max-faces',type=int,default=120000);args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();mesh=trimesh.load(args.input,force='mesh',process=False);mesh.merge_vertices(digits_vertex=7,merge_tex=True,merge_norm=True)
    mesh.update_faces(mesh.nondegenerate_faces());mesh.update_faces(mesh.unique_faces());mesh.remove_unreferenced_vertices()
    before=topology(mesh);source_mesh=mesh.copy();source=mesh.vertices.copy();extent=float(np.ptp(source,axis=0).max())
    # Tiny disconnected flecks are common decoder artifacts. Preserve structural pieces.
    components=mesh.split(only_watertight=False);total_area=mesh.area
    keep=[c for c in components if c.area>=total_area*.0003 or len(c.faces)>=300]
    if keep:mesh=trimesh.util.concatenate(keep)
    if not mesh.is_watertight and not args.voxel_only:
        vertices,faces=pymeshfix.clean_from_arrays(np.asarray(mesh.vertices),np.asarray(mesh.faces),verbose=False,joincomp=False,remove_smallest_components=False)
        mesh=trimesh.Trimesh(vertices,faces,process=False)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():raise ValueError('Repair produced invalid geometry')
    trimesh.repair.fix_normals(mesh,multibody=True)
    # MeshFix can discard open legs even when removal of small components is disabled.
    # Reject that result using both bounds and source-to-result coverage.
    sampled=source[::max(1,len(source)//15000)]
    distances_back=surface_distances(mesh,sampled)
    bounds_error=float(np.abs(mesh.bounds-source_mesh.bounds).max()/extent)
    source_retention=float((distances_back<extent*.02).mean())
    method='MeshFix with bidirectional geometry-retention checks'
    if args.voxel_only or not mesh.is_watertight or bounds_error>.025 or source_retention<.99:
        # Close a narrow voxel surface shell without dropping disconnected structural parts.
        pitch=extent/220
        vox=source_mesh.voxelized(pitch,method='subdivide')
        field=np.pad(vox.matrix,3)
        field=ndimage.binary_closing(field,structure=ndimage.generate_binary_structure(3,1),iterations=1)
        if not args.preserve_cavities:field=ndimage.binary_fill_holes(field)
        smooth=ndimage.gaussian_filter(field.astype(np.float32),.65)
        vertices,faces,_,_=marching_cubes(smooth,level=.3,spacing=(pitch,pitch,pitch),allow_degenerate=False)
        vertices+=vox.transform[:3,3]-3*pitch
        clean=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
        clean.remove_degenerate_triangles();clean.remove_duplicated_triangles();clean.remove_unreferenced_vertices()
        candidate=trimesh.Trimesh(np.asarray(clean.vertices),np.asarray(clean.triangles),process=False)
        trimesh.repair.fix_normals(candidate,multibody=True)
        if len(candidate.faces)>args.max_faces:
            reduced=clean.simplify_quadric_decimation(args.max_faces)
            trial=trimesh.Trimesh(np.asarray(reduced.vertices),np.asarray(reduced.triangles),process=False)
            if trial.is_watertight:candidate=trial
        mesh=candidate
        method='Narrow voxel solidification; extent/220 pitch, 1-voxel closing, preserved disconnected structural parts'
        if args.preserve_cavities:method+='; internal cavities retained to preserve original thin surfaces'
        bounds_error=float(np.abs(mesh.bounds-source_mesh.bounds).max()/extent)
        source_retention=float((surface_distances(mesh,sampled)<extent*.02).mean())
    if not mesh.is_watertight:
        args.output.with_suffix('.failure.json').write_text(json.dumps(topology(mesh),indent=2)+'\n')
        raise ValueError('Mesh remains non-watertight after repair')
    negligible_outlier_removed=bool(bounds_error>.05 and bounds_error<.2 and source_retention>.999)
    if (bounds_error>.05 and not negligible_outlier_removed) or source_retention<.97:raise ValueError(f'Repair lost geometry: bounds={bounds_error}, retention={source_retention}')
    after=topology(mesh);size=max(np.ptp(source,axis=0));distances=surface_distances(source_mesh,mesh.vertices[::max(1,len(mesh.vertices)//10000)])
    report={'source':str(args.input.resolve()),'before':before,'after':after,'tiny_components_removed':0 if method.startswith('Narrow voxel') else len(components)-len(keep),'tiny_components_filter_attempted':len(components)-len(keep),
            'repair':method,'distance_measure':'Exact point-to-triangle distance via Open3D RaycastingScene','source_surface_retention_at_2pct_extent':source_retention,'max_bound_change_fraction':bounds_error,'large_bound_change_only_for_negligible_outlier':negligible_outlier_removed,
            'repair_vertex_distance_p95_fraction_of_extent':float(np.quantile(distances,.95)/size),
            'seconds':time.monotonic()-start,'watertight_is_not_ground_truth_geometry':True}
    mesh.export(args.output)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
