#!/usr/bin/env python3
"""Place all independently generated/repaired objects; no background geometry."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh,torch
from scipy.spatial import cKDTree
from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdPhysics,UsdShade,Vt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'examples/mesh'))
from compose_hybrid_scene import stage_units,vertex_material
from observed_color_material import export_observed_glb,srgb_to_linear,VIEW_DOME_INTENSITY,VIEW_RENDER_SETTINGS


def save(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')

def color_vertices(directory,vertices):
    points=np.load(directory/'observed_points.npy');meta=json.loads((directory/'input/transforms.json').read_text())
    sums=np.zeros_like(points,dtype=float);weight=np.zeros(len(points));fallback=[]
    for frame in meta['frames']:
        rgba=np.asarray(Image.open(directory/'input'/frame['file_path']).convert('RGBA'))
        known=rgba[...,3]>127;fallback.append(np.median(rgba[known,:3],axis=0))
        pose=np.asarray(frame['source_camera']['c2w']);cam=(points-pose[:3,3])@pose[:3,:3]
        K=np.asarray(frame['K_image_pix']);proj=cam@K.T
        xy=np.rint(proj[:,:2]/np.maximum(proj[:,2:3],1e-9)).astype(int)
        valid=(cam[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<512)&(xy[:,1]>=0)&(xy[:,1]<512)
        idx=np.flatnonzero(valid);idx=idx[known[xy[idx,1],xy[idx,0]]]
        if len(idx):sums[idx]+=rgba[xy[idx,1],xy[idx,0],:3];weight[idx]+=1
    valid=weight>0
    if valid.sum()<10:return np.tile(np.median(fallback,axis=0)/255.,(len(vertices),1))
    color=sums[valid]/weight[valid,None]/255.
    distance,nearest=cKDTree(points[valid]).query(vertices,k=min(3,valid.sum()),workers=4)
    if distance.ndim==1:return color[nearest]
    w=1/np.maximum(distance,meta['scale']*.002);w/=w.sum(1,keepdims=True)
    return (color[nearest]*w[...,None]).sum(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'results/scene_gen_room/full_objects');a=p.parse_args()
    root=a.root;out=root/'scene';assets=out/'assets';assets.mkdir(parents=True,exist_ok=True)
    objects=json.loads((root/'inventory.json').read_text())['objects'];choices=json.loads((root/'selected_meshes.json').read_text())['objects']
    choice={o['instance_id']:o for o in choices};assert set(choice)=={o['instance_id'] for o in objects},'Incomplete object selection'
    coordinates=json.loads((ROOT/'results/scene_gen_room/simulation/coordinates.json').read_text());T=np.asarray(coordinates['world_to_simulation'])
    stage=Usd.Stage.CreateNew(str(assets/'objects.usdc'));stage_units(stage);x=UsdGeom.Xform.Define(stage,'/Objects');stage.SetDefaultPrim(x.GetPrim());material=vertex_material(stage)
    glb=trimesh.Scene();records=[];all_bounds=[];z2y=np.array([[1.,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    for obj in objects:
        ident=obj['instance_id'];selected=choice[ident];directory=root/obj['directory'];path=root/selected['repaired_mesh']
        mesh=trimesh.load(path,force='mesh',process=False)
        if not mesh.is_watertight or not np.isfinite(mesh.vertices).all():raise ValueError(f'Invalid final mesh {path}')
        original=mesh.vertices.copy();colors=color_vertices(directory,original);mesh.visual.vertex_colors=np.column_stack([colors*255,np.full(len(colors),255)]).astype(np.uint8)
        mesh.apply_transform(T);placed=mesh.vertices.copy();center=T[:3,:3]@np.asarray(obj['offset'])+T[:3,3];mesh.vertices-=center
        name=obj['name'].replace(' ','_');parent=UsdGeom.Xform.Define(stage,f'/Objects/{name}');parent.AddTranslateOp().Set(Gf.Vec3d(*center))
        parent.GetPrim().SetCustomDataByKey('instanceId',ident);parent.GetPrim().SetCustomDataByKey('category',obj['category']);parent.GetPrim().SetCustomDataByKey('geometrySource',selected['source_mesh'])
        visual=UsdGeom.Mesh.Define(stage,f'/Objects/{name}/Visual');visual.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices,np.float32)))
        visual.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces),3,np.int32)));visual.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces,np.int32).ravel()))
        visual.CreateSubdivisionSchemeAttr('none');visual.CreateDoubleSidedAttr(True);visual.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.bounds,np.float32)))
        visual.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_normals,np.float32)));visual.SetNormalsInterpolation('vertex')
        linear=srgb_to_linear(colors);visual.CreateDisplayColorPrimvar('vertex').Set(Vt.Vec3fArray.FromNumpy(linear.astype(np.float32)))
        UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material);UsdPhysics.CollisionAPI.Apply(visual.GetPrim()).CreateCollisionEnabledAttr(True);UsdPhysics.MeshCollisionAPI.Apply(visual.GetPrim()).CreateApproximationAttr('none')
        exported=mesh.copy();exported.vertices=placed;exported.apply_transform(z2y);export_observed_glb(exported,assets/f'{name}.glb');glb.add_geometry(exported,node_name=name,geom_name=name)
        inv=np.linalg.inv(T);roundtrip=placed@inv[:3,:3].T+inv[:3,3];error=float(np.abs(roundtrip-original).max());assert error<1e-6
        bound=[placed.min(0).tolist(),placed.max(0).tolist()];all_bounds.append(bound)
        record={**selected,'name':name,'category':obj['category'],'placement_method':obj.get('bounds_method'),'source_segmentation_views':obj['views'],'faces':len(mesh.faces),'vertices':len(mesh.vertices),'watertight':True,'position_m':center.tolist(),'bounds_m':bound,'placement_roundtrip_max_error':error,'asset':f'assets/{name}.glb','color_source':'Nearest multiview observed surface colors; inferred back-side colors'}
        records.append(record);print(name,len(mesh.faces),'triangles',flush=True)
    stage.GetRootLayer().Save();export_observed_glb(glb,out/'scene.glb')
    scene=Usd.Stage.CreateNew(str(out/'scene.usda'));stage_units(scene);world=UsdGeom.Xform.Define(scene,'/World');scene.SetDefaultPrim(world.GetPrim());UsdGeom.Xform.Define(scene,'/World/Objects').GetPrim().GetReferences().AddReference('assets/objects.usdc')
    physics=UsdPhysics.Scene.Define(scene,'/World/PhysicsScene');physics.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));physics.CreateGravityMagnitudeAttr(9.81)
    bounds=np.array([np.asarray(all_bounds)[:,0].min(0),np.asarray(all_bounds)[:,1].max(0)])
    floor=UsdGeom.Cube.Define(scene,'/World/FloorSupport');floor.CreateSizeAttr(1);floor.AddTranslateOp().Set(Gf.Vec3d(*bounds.mean(0)[:2],-.025));floor.AddScaleOp().Set(Gf.Vec3f(*(bounds[1]-bounds[0])[:2]+1,.05));floor.CreateVisibilityAttr('invisible');UsdPhysics.CollisionAPI.Apply(floor.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdLux.DomeLight.Define(scene,'/World/Lighting').CreateIntensityAttr(VIEW_DOME_INTENSITY)
    views=torch.load(ROOT/'data/mipnerf360_indoor/room/cluster_result/gauscluster_tracking_data.pt',map_location='cpu',weights_only=False)['view_data']
    for index in [0,60,120,180,200,240,270,300]:
        v=views[index];pose=v['camtoworld'].numpy().copy();pose[:3,3]=T[:3,:3]@pose[:3,3]+T[:3,3];pose[:3,:3]=T[:3,:3]/coordinates['meters_per_scene_unit']@pose[:3,:3];pose=pose@np.diag([1.,-1.,-1.,1.])
        camera=UsdGeom.Camera.Define(scene,f'/World/Cameras/RecordedView_{index}');camera.AddTransformOp().Set(Gf.Matrix4d(pose.T.tolist()));camera.CreateHorizontalApertureAttr(36.);camera.CreateVerticalApertureAttr(36*v['height']/v['width']);camera.CreateFocalLengthAttr(float(v['K'][0,0])*36/v['width']);camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100.))
    center=bounds.mean(0);z=np.array([0.,-.55,.835]);z/=np.linalg.norm(z);x=np.cross([0.,0.,1.],z);x/=np.linalg.norm(x);y=np.cross(z,x);corners=np.array([[a,b,c] for a in bounds[:,0] for b in bounds[:,1] for c in bounds[:,2]])-center;distance=1.12*max(np.max(np.abs(corners@x)/.6+corners@z),np.max(np.abs(corners@y)/.4+corners@z));eye=center+z*distance;pose=np.eye(4);pose[:3,:3]=np.column_stack([x,y,z]);pose[:3,3]=eye
    camera=UsdGeom.Camera.Define(scene,'/World/Cameras/Overview');camera.AddTransformOp().Set(Gf.Matrix4d(pose.T.tolist()));camera.CreateHorizontalApertureAttr(36.);camera.CreateVerticalApertureAttr(24.);camera.CreateFocalLengthAttr(30.);camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100.))
    scene.GetRootLayer().customLayerData={'sceneMode':'generated_object_meshes_only','expectedMeshCount':len(objects),'expectedColliderCount':len(objects)+1,'expectedGaussianVolumeCount':0,'selectionRoot':'/World/Objects','defaultCamera':'/World/Cameras/Overview','renderSettings':VIEW_RENDER_SETTINGS,'appearanceMode':'photo_vertex_colors_with_lit_dielectric_material'}
    scene.GetRootLayer().Save();check=Usd.Stage.Open(str(out/'scene.usda'));prims=list(check.Traverse());meshes=[p for p in prims if p.IsA(UsdGeom.Mesh)];colliders=[p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(meshes)==len(objects) and len(colliders)==len(objects)+1
    assert all(str(p.GetPath()).startswith('/World/Objects/') for p in meshes)
    assert not any(p.IsA(UsdGeom.Points) or p.GetTypeName()=='Volume' for p in prims)
    checkglb=trimesh.load(out/'scene.glb',force='scene',process=False);assert len(checkglb.geometry)==len(objects)
    manifest={'object_count':len(objects),'mesh_count':len(meshes),'collider_count':len(colliders),'triangles':sum(r['faces'] for r in records),'watertight_meshes':sum(r['watertight'] for r in records),'gaussian_volume_count':0,'point_cloud_count':0,'background_mesh_count':0,'usd_units':'meters, Z up','glb_units':'meters, Y up','scale_is_estimated':True,'meters_per_original_scene_unit':coordinates['meters_per_scene_unit'],'physics':'Independent static triangle colliders and invisible floor support; no mass/joint/dynamic-object certification','objects':records,'world_to_simulation':T.tolist(),'bounds_m':bounds.tolist()}
    save(out/'manifest.json',manifest);print('ASSEMBLED',len(objects),'objects',flush=True)
if __name__=='__main__':main()
