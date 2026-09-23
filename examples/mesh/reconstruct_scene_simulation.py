#!/usr/bin/env python3
"""Fuse calibrated Gaussian depth into a colored, metric, Z-up room mesh.

The metric scale is estimated from an explicitly supplied chair height unless a
measured scale is provided. Only observed geometry is reconstructed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import cv2
import torch
import open3d as o3d

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'examples/segmentation'))
from generate_instance_views import load_gaussians_from_ckpt
from gsplat.rendering import rasterization


def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def camera_points(depth,K,c2w,stride=7):
    y,x=np.mgrid[0:depth.shape[0]:stride,0:depth.shape[1]:stride]
    z=depth[::stride,::stride];valid=z>0
    cam=np.stack([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z],axis=-1)[valid]
    return cam@c2w[:3,:3].T+c2w[:3,3]


def render_cache(args,views):
    cache=args.output/'cache';cache.mkdir(parents=True,exist_ok=True)
    settings={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_size':args.checkpoint.stat().st_size,
              'checkpoint_mtime_ns':args.checkpoint.stat().st_mtime_ns,'alpha_min':args.alpha_min,
              'max_depth_scene_units':args.max_depth,'views':len(views),'depth_edge_filter':'max(0.015, 1.5 percent depth); one-pixel neighborhood'}
    old=cache/'render_settings.json'
    if old.exists() and json.loads(old.read_text())!=settings:raise ValueError('Render cache settings changed; choose a new output directory')
    save_json(old,settings)
    missing=[v for v in views if not(cache/(v['image_name']+'.npz')).exists()]
    if not missing:return
    gs=load_gaussians_from_ckpt(str(args.checkpoint),torch.device('cuda'))
    means,quats,scales,opacity,colors,degree=gs
    for number,v in enumerate(missing):
        path=cache/(v['image_name']+'.npz');c2w=v['camtoworld'].numpy();K=v['K'].numpy()
        with torch.no_grad():
            rc,alpha,meta=rasterization(means=means,quats=quats,scales=scales,opacities=opacity,colors=colors,
                viewmats=torch.linalg.inv(v['camtoworld'].cuda())[None],Ks=v['K'].cuda()[None],
                width=v['width'],height=v['height'],sh_degree=degree,render_mode='RGB+ED',packed=False)
        median=meta.get('render_median')
        used_median=isinstance(median,torch.Tensor) and median.numel()>0
        depth=(median[0,...,0] if used_median else rc[0,...,-1]).cpu().numpy()
        a=alpha[0,...,0].cpu().numpy()
        valid=np.isfinite(depth)&(depth>.05)&(depth<args.max_depth)&(a>=args.alpha_min)
        high=cv2.dilate(depth,np.ones((3,3),np.uint8));low=cv2.erode(depth,np.ones((3,3),np.uint8))
        valid&=(high-low)<np.maximum(.015,depth*.015)
        depth[~valid]=0
        tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,depth=depth.astype(np.float32),median_depth=used_median)
        tmp.replace(path)
        if number%20==0 or number==len(missing)-1:print(f'Render {number+1}/{len(missing)} {v["image_name"]}: valid {valid.mean():.1%}',flush=True)
    del gs,means,quats,scales,opacity,colors,rc,alpha,meta
    torch.cuda.empty_cache()


def fit_coordinates(args,views):
    cache=args.output/'cache'
    pieces=[camera_points(np.load(cache/(v['image_name']+'.npz'))['depth'],v['K'].numpy(),v['camtoworld'].numpy(),9) for v in views[::4]]
    points=np.concatenate(pieces)
    pcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points));pcd=pcd.voxel_down_sample(.012)
    # Reject isolated depth artifacts before fitting structural planes.
    pcd,_=pcd.remove_statistical_outlier(nb_neighbors=24,std_ratio=2.5)
    remaining=pcd
    cameras=np.stack([v['camtoworld'].numpy() for v in views]);min_camera_z=float(np.quantile(cameras[:,2,3],.05))
    candidates=[]
    for i in range(10):
        if len(remaining.points)<1000:break
        plane,ids=remaining.segment_plane(distance_threshold=.008,ransac_n=3,num_iterations=1500)
        plane=np.asarray(plane)
        if plane[2]<0:plane=-plane
        pp=np.asarray(remaining.points)[ids]
        center=np.median(pp,axis=0);span=np.quantile(pp,.98,axis=0)-np.quantile(pp,.02,axis=0)
        floor_candidate=plane[2]>.94 and center[2]<min_camera_z-.18
        candidates.append({'plane':plane.tolist(),'inliers':len(ids),'median':center.tolist(),'span':span.tolist(),'floor_candidate':bool(floor_candidate)})
        remaining=remaining.select_by_index(ids,invert=True)
    eligible=[p for p in candidates if p['floor_candidate']]
    if not eligible:raise RuntimeError('No well-supported horizontal floor plane found')
    floor=max(eligible,key=lambda p:p['inliers']);normal=np.array(floor['plane'][:3]);normal/=np.linalg.norm(normal)
    z=np.array([0.,0.,1.]);cross=np.cross(normal,z);c=float(normal@z)
    skew=np.array([[0,-cross[2],cross[1]],[cross[2],0,-cross[0]],[-cross[1],cross[0],0]])
    rotation=np.eye(3)+skew+skew@skew/(1+c)
    floor_point=-np.array(floor['plane'][:3])*floor['plane'][3]
    chair_height=None
    if args.meters_per_scene_unit is not None:
        if not np.isfinite(args.meters_per_scene_unit) or args.meters_per_scene_unit<=0:
            raise ValueError('meters-per-scene-unit must be positive and finite')
        scale=args.meters_per_scene_unit
    else:
        if not args.chair_mesh.is_file():
            raise FileNotFoundError('Provide --chair-mesh for estimated scale, or a measured --meters-per-scene-unit')
        import trimesh
        chair=trimesh.load(args.chair_mesh,force='mesh',process=False)
        rotated=(np.asarray(chair.vertices)-floor_point)@rotation.T
        chair_height=float(np.quantile(rotated[:,2],.995))
        if chair_height<=0 or args.chair_height_m<=0:raise ValueError('Chair height must be positive')
        scale=args.chair_height_m/chair_height
    # Place the horizontal origin near the camera trajectory, projected on floor.
    center=np.median(cameras[:,:3,3],axis=0);center-=normal*(normal@center+floor['plane'][3])
    matrix=np.eye(4);matrix[:3,:3]=scale*rotation;matrix[:3,3]=-scale*(rotation@center)
    transformed=points@matrix[:3,:3].T+matrix[:3,3]
    # Observed bounds are robust to sparse reflections / points through windows.
    lower=np.quantile(transformed,.002,axis=0);upper=np.quantile(transformed,.998,axis=0)
    lower[2]=-.08;upper[2]=min(float(upper[2]),3.2)
    report={'world_to_simulation':matrix.tolist(),'simulation_to_world':np.linalg.inv(matrix).tolist(),
            'meters_per_scene_unit':scale,'scale_is_estimated':args.meters_per_scene_unit is None,
            'scale_assumption':f'chair top above fitted floor = {args.chair_height_m} m' if args.meters_per_scene_unit is None else 'user supplied meters per normalized scene unit',
            'chair_height_scene_units':chair_height,'floor_plane_world':floor['plane'],'plane_candidates':candidates,
            'bounds_simulation_m':[lower.tolist(),upper.tolist()],'up_axis':'Z','meters_per_unit':1.0,
            'floor_fit_threshold_scene_units':.008,'floor_fit_points':len(pcd.points)}
    save_json(args.output/'coordinates.json',report)
    print('Coordinate fit',json.dumps({k:report[k] for k in ['meters_per_scene_unit','floor_plane_world','bounds_simulation_m']}),flush=True)
    return report


def fuse(args,views,coordinates):
    matrix=np.array(coordinates['world_to_simulation']);scale=coordinates['meters_per_scene_unit'];rot=matrix[:3,:3]/scale
    bounds=np.array(coordinates['bounds_simulation_m'])
    volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=args.voxel_m,sdf_trunc=args.voxel_m*4,
                                                       color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i,v in enumerate(views):
        depth=np.load(args.output/'cache'/(v['image_name']+'.npz'))['depth']*scale
        color=np.asarray(Image.open(v['image_path']).convert('RGB'))
        h,w=depth.shape;K=v['K'].numpy()
        pose=v['camtoworld'].numpy().copy();pose[:3,3]=matrix[:3,:3]@pose[:3,3]+matrix[:3,3];pose[:3,:3]=rot@pose[:3,:3]
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color),o3d.geometry.Image(depth.astype(np.float32)),depth_scale=1.,depth_trunc=args.max_depth*scale,convert_rgb_to_intensity=False)
        intrinsic=o3d.camera.PinholeCameraIntrinsic(w,h,float(K[0,0]),float(K[1,1]),float(K[0,2]),float(K[1,2]))
        volume.integrate(rgbd,intrinsic,np.linalg.inv(pose))
        if i%30==0:print(f'TSDF fusion {i+1}/{len(views)}',flush=True)
    mesh=volume.extract_triangle_mesh()
    raw_count=len(mesh.triangles)
    mesh=mesh.crop(o3d.geometry.AxisAlignedBoundingBox(bounds[0],bounds[1]))
    mesh.remove_duplicated_vertices();mesh.remove_duplicated_triangles();mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices()
    clusters,counts,areas=mesh.cluster_connected_triangles();clusters=np.asarray(clusters);counts=np.asarray(counts);areas=np.asarray(areas)
    remove=(counts[clusters]<60)|(areas[clusters]<.002)
    mesh.remove_triangles_by_mask(remove);mesh.remove_unreferenced_vertices()
    # Slight smoothing suppresses voxel stair steps without closing doors or gaps.
    mesh=mesh.filter_smooth_taubin(number_of_iterations=2)
    if len(mesh.triangles)>args.faces:mesh=mesh.simplify_quadric_decimation(args.faces)
    mesh.remove_duplicated_vertices();mesh.remove_duplicated_triangles();mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors():raise ValueError('TSDF produced no visual colors')
    o3d.io.write_triangle_mesh(str(args.output/'scene_observed.ply'),mesh,write_ascii=False)
    result={'status':'succeeded','views':len(views),'voxel_m':args.voxel_m,'sdf_trunc_m':args.voxel_m*4,'raw_faces':raw_count,
            'faces':len(mesh.triangles),'vertices':len(mesh.vertices),'bounds_m':np.stack([mesh.get_min_bound(),mesh.get_max_bound()]).tolist(),
            'is_watertight':mesh.is_watertight(),'vertex_manifold':mesh.is_vertex_manifold(),'edge_manifold_without_boundary':mesh.is_edge_manifold(allow_boundary_edges=False),
            'source':'original room 3DGS median/expected depth and real RGB, all 311 calibrated views',
            'scope':'observed surfaces; unseen areas are not invented; static environment candidate'}
    save_json(args.output/'reconstruction.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/scene_gen_room/simulation')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'results/scene_gen_room/ckpts/ckpt_6999_rank0.pt')
    p.add_argument('--tracking',type=Path,default=ROOT/'data/mipnerf360_indoor/room/cluster_result/gauscluster_tracking_data.pt')
    p.add_argument('--stage',choices=['all','render','coordinates','fuse'],default='all')
    p.add_argument('--alpha-min',type=float,default=.95)
    p.add_argument('--max-depth',type=float,default=5.)
    p.add_argument('--chair-height-m',type=float,default=1.)
    p.add_argument('--chair-mesh',type=Path,default=ROOT/'results/scene_gen_room/multiview_comparison/direct/instance_0/reconstruction_train/mesh_world.glb')
    p.add_argument('--meters-per-scene-unit',type=float)
    p.add_argument('--voxel-m',type=float,default=.012)
    p.add_argument('--faces',type=int,default=900000)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);o3d.utility.random.seed(42)
    views=torch.load(args.tracking,map_location='cpu',weights_only=False)['view_data']
    if args.stage in ['all','render']:render_cache(args,views)
    if args.stage in ['all','coordinates']:coordinates=fit_coordinates(args,views)
    else:
        path=args.output/'coordinates.json';coordinates=json.loads(path.read_text()) if path.exists() else None
    if args.stage in ['all','fuse']:
        if coordinates is None:raise ValueError('Run coordinates stage first')
        fuse(args,views,coordinates)

if __name__=='__main__':main()
