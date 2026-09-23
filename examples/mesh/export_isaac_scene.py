#!/usr/bin/env python3
"""Package an observed room mesh with separate static collision meshes for Isaac Sim.

Visuals use vertex colors. Collision shells are closed voxel surfaces, preserving
room concavity. All scene parts are static; no mass or movable-joint claims.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy import ndimage
from skimage.measure import marching_cubes
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,UsdLux,Sdf,Gf,Vt

ROOT=Path(__file__).resolve().parents[2]


def save_json(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def simplify(mesh,faces):
    if len(mesh.faces)<=faces:return mesh.copy()
    m=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces))
    m.vertex_colors=o3d.utility.Vector3dVector(mesh.visual.vertex_colors[:,:3]/255)
    m=m.simplify_quadric_decimation(faces)
    m.remove_degenerate_triangles();m.remove_duplicated_triangles();m.remove_unreferenced_vertices()
    return trimesh.Trimesh(vertices=np.asarray(m.vertices),faces=np.asarray(m.triangles),
                           vertex_colors=np.clip(np.asarray(m.vertex_colors)*255,0,255).astype(np.uint8),process=False)


def split_parts(mesh,coordinates):
    ckpt=torch.load(ROOT/'results/scene_gen_room/ckpts/ckpt_6999_rank0.pt',map_location='cpu',weights_only=True)['splats']
    means=ckpt['means'].numpy();labels=np.load(ROOT/'data/mipnerf360_indoor/room/cluster_result/instance_labels_dense.npy')
    matrix=np.array(coordinates['simulation_to_world']);query=mesh.vertices@matrix[:3,:3].T+matrix[:3,3]
    tree=cKDTree(means);distance,ids=tree.query(query,k=3,workers=8)
    nearest=labels[ids].copy();nearest[distance>.03]=-1
    vertex_labels=np.full(len(mesh.vertices),-1,np.int32)
    for ident in [0,1,2]:vertex_labels[(nearest==ident).sum(1)>=2]=ident
    vertex_labels[mesh.vertices[:,2]<.045]=-1
    face_labels=np.full(len(mesh.faces),-1,np.int32)
    for ident in [0,1,2]:face_labels[(vertex_labels[mesh.faces]==ident).sum(1)>=2]=ident
    names={-1:'Background',0:'Armchair',1:'SideTable',2:'Footstool'}
    parts={}
    for ident,name in names.items():
        mask=face_labels==ident
        if mask.any():parts[name]=mesh.submesh([mask],append=True,repair=False)
    assert sum(len(m.faces) for m in parts.values())==len(mesh.faces)
    return parts


def collision_shell(mesh,pitch,remove_floor=False,cache_path=None):
    source=mesh.copy()
    if remove_floor:
        source.update_faces(source.triangles[:,:,2].max(1)>.06);source.remove_unreferenced_vertices()
    source=simplify(source,180000)
    vox=source.voxelized(pitch,method='subdivide')
    matrix=np.pad(vox.matrix,3)
    # Close only gaps on the order of one voxel. Never fill the room interior.
    matrix=ndimage.binary_closing(matrix,structure=ndimage.generate_binary_structure(3,1),iterations=1)
    labels,count=ndimage.label(matrix)
    sizes=np.bincount(labels.ravel());matrix&=sizes[labels]>=5
    # A smooth scalar field avoids edge-touching binary shells becoming non-manifold.
    field=ndimage.gaussian_filter(matrix.astype(np.float32),sigma=.65)
    origin=vox.transform[:3,3]-3*pitch
    if cache_path is not None:np.savez_compressed(cache_path,field=field,origin=origin,pitch=pitch)
    vertices,faces,_,_=marching_cubes(field,level=.3,spacing=(pitch,pitch,pitch),allow_degenerate=False)
    vertices+=origin
    clean=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
    clean.merge_close_vertices(.00002)
    clean.remove_degenerate_triangles();clean.remove_duplicated_triangles();clean.remove_unreferenced_vertices()
    # Validate the actual float32 coordinates written into PLY and USD.
    result=trimesh.Trimesh(vertices=np.asarray(clean.vertices).astype(np.float32),faces=np.asarray(clean.triangles),process=False)
    if (result.area_faces<=1e-12).any():raise ValueError('Collision contains degenerate float32 triangles')
    trimesh.repair.fix_normals(result,multibody=True)
    if not np.isfinite(result.vertices).all() or not result.is_watertight:raise ValueError('Collision shell is not finite and watertight')
    return result


def srgb_linear(rgb):return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


def author_mesh(stage,path,mesh,visual=False,collision=False,material=None):
    prim=UsdGeom.Mesh.Define(stage,path)
    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices,dtype=np.float32)))
    prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces),3,dtype=np.int32)))
    prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces,dtype=np.int32).reshape(-1)))
    prim.CreateSubdivisionSchemeAttr('none');prim.CreateDoubleSidedAttr(True);prim.CreateOrientationAttr('rightHanded')
    prim.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.bounds,dtype=np.float32)))
    if visual:
        colors=srgb_linear(np.asarray(mesh.visual.vertex_colors[:,:3],dtype=np.float32)/255)
        prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(colors))
        prim.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_normals,dtype=np.float32)))
        prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        if material:UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr('none')
        prim.CreatePurposeAttr(UsdGeom.Tokens.guide)
        prim.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    return prim


def export_usd(args,parts,colliders,coordinates,manifest):
    assets=args.output/'assets';assets.mkdir(exist_ok=True)
    layer=Usd.Stage.CreateNew(str(assets/'room_meshes.usdc'))
    UsdGeom.SetStageUpAxis(layer,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(layer,1.)
    UsdPhysics.SetStageKilogramsPerUnit(layer,1.)
    root=UsdGeom.Xform.Define(layer,'/Room');layer.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey('source','Mip-NeRF 360 room: 311-view TSDF fusion')
    root.GetPrim().SetCustomDataByKey('scaleEstimated',coordinates['scale_is_estimated'])
    root.GetPrim().SetCustomDataByKey('metersPerSceneUnit',coordinates['meters_per_scene_unit'])
    material=UsdShade.Material.Define(layer,'/Room/Materials/Photogrammetry')
    shader=UsdShade.Shader.Define(layer,'/Room/Materials/Photogrammetry/Surface');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.9);shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
    reader=UsdShade.Shader.Define(layer,'/Room/Materials/Photogrammetry/VertexColor');reader.CreateIdAttr('UsdPrimvarReader_float3')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('displayColor');reader.CreateOutput('result',Sdf.ValueTypeNames.Float3)
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(),'result')
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    physics_material=UsdShade.Material.Define(layer,'/Room/Materials/Contact')
    mat=UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim());mat.CreateStaticFrictionAttr(.7);mat.CreateDynamicFrictionAttr(.6);mat.CreateRestitutionAttr(.02)
    for name,mesh in parts.items():
        UsdGeom.Xform.Define(layer,f'/Room/{name}')
        author_mesh(layer,f'/Room/{name}/Visual',mesh,visual=True,material=material)
        if name in colliders:
            collider=author_mesh(layer,f'/Room/{name}/Collision',colliders[name],collision=True)
            UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(physics_material,materialPurpose='physics')
    # A bounded fitted floor is explicit inferred support, not a scanned solid.
    b=np.array(manifest['floor_bounds_m']);size=b[1]-b[0];center=b.mean(0)
    floor=UsdGeom.Cube.Define(layer,'/Room/FloorSupport');floor.CreateSizeAttr(1.)
    xf=UsdGeom.Xformable(floor);xf.AddTranslateOp().Set(Gf.Vec3d(*center));xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim()).CreateCollisionEnabledAttr(True)
    floor.CreatePurposeAttr(UsdGeom.Tokens.guide);floor.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    floor.GetPrim().SetCustomDataByKey('inferredSupport',True)
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(physics_material,materialPurpose='physics')
    layer.GetRootLayer().Save()
    stage=Usd.Stage.CreateNew(str(args.output/'scene.usda'));UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.);UsdPhysics.SetStageKilogramsPerUnit(stage,1.)
    world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim())
    scene=UsdGeom.Xform.Define(stage,'/World/Room');scene.GetPrim().GetReferences().AddReference('assets/room_meshes.usdc')
    physics=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene');physics.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));physics.CreateGravityMagnitudeAttr(9.81)
    dome=UsdLux.DomeLight.Define(stage,'/World/Lighting');dome.CreateIntensityAttr(400.)
    # Named measured cameras make inspecting the imported room straightforward.
    source_views=torch.load(ROOT/'data/mipnerf360_indoor/room/cluster_result/gauscluster_tracking_data.pt',map_location='cpu',weights_only=False)['view_data']
    transform=np.array(coordinates['world_to_simulation']);metric_scale=coordinates['meters_per_scene_unit']
    for index in [0,120,200,270]:
        view=source_views[index];pose=view['camtoworld'].numpy().copy()
        pose[:3,3]=transform[:3,:3]@pose[:3,3]+transform[:3,3]
        pose[:3,:3]=transform[:3,:3]/metric_scale@pose[:3,:3]
        pose=pose@np.diag([1.,-1.,-1.,1.])
        camera=UsdGeom.Camera.Define(stage,f'/World/Cameras/RecordedView_{index}')
        UsdGeom.Xformable(camera).AddTransformOp().Set(Gf.Matrix4d(pose.T.tolist()))
        camera.CreateHorizontalApertureAttr(36.);camera.CreateVerticalApertureAttr(36.*view['height']/view['width'])
        camera.CreateFocalLengthAttr(float(view['K'][0,0])*36./view['width']);camera.CreateClippingRangeAttr(Gf.Vec2f(.02,100.))
    stage.SetStartTimeCode(0);stage.SetEndTimeCode(600);stage.SetTimeCodesPerSecond(60);stage.SetFramesPerSecond(60)
    stage.GetRootLayer().customLayerData={'notes':'Static reconstructed environment. Estimated metric scale; approximate voxel-shell collisions; see manifest.json.'}
    stage.GetRootLayer().Save()
    # Round-trip structure verification, before any simulator-side physics test.
    check=Usd.Stage.Open(str(args.output/'scene.usda'))
    paths=[str(p.GetPath()) for p in check.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    if len(paths)!=len(colliders)+1:raise ValueError('USD collider count mismatch')
    if not check.GetDefaultPrim() or UsdGeom.GetStageMetersPerUnit(check)!=1 or UsdGeom.GetStageUpAxis(check)!='Z':raise ValueError('USD coordinate metadata missing')
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/scene_gen_room/simulation')
    p.add_argument('--collision-voxel-m',type=float,default=.025)
    args=p.parse_args();assets=args.output/'assets';assets.mkdir(parents=True,exist_ok=True)
    coordinates=json.loads((args.output/'coordinates.json').read_text())
    mesh=trimesh.load(args.output/'scene_observed.ply',force='mesh',process=False)
    parts=split_parts(mesh,coordinates)
    save_json(assets/'parts.json',{name:{'faces':len(m.faces),'vertices':len(m.vertices),'bounds_m':m.bounds.tolist(),'source':'observed multiview TSDF'} for name,m in parts.items()})
    colliders={};stats=[]
    for name,part in parts.items():
        print(f'Part {name}: {len(part.faces)} visual faces',flush=True)
        part.export(assets/(name.lower()+'_visual.ply'))
        collision=collision_shell(part,args.collision_voxel_m,remove_floor=name=='Background',cache_path=args.output/'cache'/(name.lower()+'_collision_field.npz'))
        collision.export(assets/(name.lower()+'_collision.ply'))
        # Portable Z-up collision OBJ is useful in other engines, too.
        collision.export(assets/(name.lower()+'_collision.obj'),include_color=False)
        colliders[name]=collision
        stats.append({'part':name,'visual_faces':len(part.faces),'collision_faces':len(collision.faces),'collision_watertight':bool(collision.is_watertight),
                      'collision_winding_consistent':bool(collision.is_winding_consistent),'collision_bounds_m':collision.bounds.tolist()})
        print(f'  Collision: {len(collision.faces)} faces, watertight={collision.is_watertight}',flush=True)
    bounds=mesh.bounds.copy();floor_bounds=bounds.copy();floor_bounds[0,2]=-.12;floor_bounds[1,2]=0
    floor_bounds[0,:2]-=.03;floor_bounds[1,:2]+=.03
    manifest={'scope':'Static environment for Isaac Sim; furniture separated but not dynamic or articulated',
        'visual_source':'all 311 calibrated views of original room GS, not the lower-quality API-generated replacement meshes',
        'meters_per_unit':1,'up_axis':'Z','scale_is_estimated':coordinates['scale_is_estimated'],'coordinate_record':'coordinates.json',
        'collision_method':'2.5-cm occupied-surface voxel shells; one local binary-closing pass; Gaussian sigma=.65 voxels, isolevel=.3; no convex hull of whole room and no filling room interior',
        'collision_voxel_m':args.collision_voxel_m,'nominal_collision_tolerance_m':2*args.collision_voxel_m,'collision_weld_m':.00002,
        'floor_bounds_m':floor_bounds.tolist(),'floor_support_is_inferred':True,
        'friction_is_assumed':True,'static_friction':.7,'dynamic_friction':.6,'restitution':.02,
        'parts':stats,'isaac_runtime_validation':'pending','limitations':['unseen geometry is absent','metric scale estimated from 1 m chair height','voxel collisions unsuitable for precision manipulation','all scene geometry static; no estimated masses or joints'],
        'entrypoint':'scene.usda','geometry_layer':'assets/room_meshes.usdc','glb':'scene.glb','glb_up_axis':'Y'}
    manifest['usd_collision_paths']=export_usd(args,parts,colliders,coordinates,manifest)
    # glTF uses Y-up. Bake a proper rotation, not an axis reflection.
    glb_transform=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]],dtype=float)
    glb=trimesh.Scene()
    for name,part in parts.items():
        part=part.copy();part.apply_transform(glb_transform);glb.add_geometry(part,node_name=name,geom_name=name)
    glb.export(args.output/'scene.glb')
    loaded=trimesh.load(args.output/'scene.glb',force='scene')
    assert sum(len(g.faces) for g in loaded.geometry.values())==len(mesh.faces)
    manifest['glb_from_simulation']=glb_transform.tolist();manifest['glb_roundtrip_faces']=len(mesh.faces)
    save_json(args.output/'manifest.json',manifest)
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
