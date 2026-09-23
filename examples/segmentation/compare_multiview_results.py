#!/usr/bin/env python3
"""Summarize paired real-view and completed-view mesh runs, with visual evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh

ROOT=Path(__file__).resolve().parents[2]


def topology(path):
    mesh=trimesh.load(path,force='mesh',process=False)
    # GLB may duplicate positions for normals/material seams: weld identical
    # vertices before inspecting topology, without moving their coordinates.
    mesh.merge_vertices(digits_vertex=7,merge_tex=True,merge_norm=True)
    faces=np.asarray(mesh.faces)
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    graph=coo_matrix((np.ones(len(unique)),(unique[:,0],unique[:,1])),shape=(len(mesh.vertices),len(mesh.vertices))).tocsr()
    _,labels=connected_components(graph,directed=False)
    face_labels=labels[faces[:,0]]
    groups=np.bincount(face_labels)
    return {'faces':len(faces),'welded_vertices':len(mesh.vertices),
            'boundary_edges':int((counts==1).sum()),'nonmanifold_edges':int((counts>2).sum()),
            'boundary_edge_fraction':float((counts==1).mean()),
            'components_with_at_least_25_faces':int((groups>=25).sum()),
            'largest_component_face_fraction':float(groups.max()/len(faces)),
            'watertight_after_welding':bool((counts==2).all()),
            'note':'Topology proxies only; watertightness and connectedness do not establish geometric accuracy.'}


def font(size=22):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'results/scene_gen_room/multiview_comparison')
    p.add_argument('--sampler',choices=['official','train'],default='official')
    p.add_argument('--instances',type=int,nargs='+',default=[0,1])
    args=p.parse_args();name='reconstruction' if args.sampler=='official' else 'reconstruction_train'
    pairs=[]
    w=384;h=384;label_h=44
    overview=Image.new('RGB',(w*4,(h+label_h)*len(args.instances)+64),'white')
    draw=ImageDraw.Draw(overview)
    draw.text((12,12),f'Multi-view mesh comparison | sampler={args.sampler} | seed=42',font=font(23),fill='black')
    for row,ident in enumerate(args.instances):
        direct=args.root/'direct'/f'instance_{ident}'
        completed=args.root/'completed'/f'instance_{ident}'
        before=json.loads((direct/name/'inspection.json').read_text())
        after=json.loads((completed/name/'inspection.json').read_text())
        drun=json.loads((direct/name/'run.json').read_text())
        arun=json.loads((completed/name/'run.json').read_text())
        for key in ['seed','sampler','resolution','geometry_only','glb_face_budget','code_commit']:
            if drun[key]!=arun[key]:raise ValueError(f'Unmatched paired setting: {key}')
        dmeta=json.loads((direct/'transforms.json').read_text())
        ameta=json.loads((completed/'transforms.json').read_text())
        if dmeta['frames']!=ameta['frames']:raise ValueError('Comparison cameras differ')
        pair={'instance_id':ident,'category':before['category'],'sampler':args.sampler,
              'direct_visible_iou':before['mean_visible_mask_iou'],'completed_visible_iou':after['mean_visible_mask_iou'],
              'iou_delta':after['mean_visible_mask_iou']-before['mean_visible_mask_iou'],
              'direct_topology':topology(direct/name/'mesh_world.glb'),
              'completed_topology':topology(completed/name/'mesh_world.glb'),
              'direct_glb':str(direct/name/'mesh_world.glb'),'completed_glb':str(completed/name/'mesh_world.glb')}
        pair['completion_method']=ameta.get('completion',{}).get('method','strict sheet extraction')
        if (direct/name/'additional_views.json').exists() and (completed/name/'additional_views.json').exists():
            extra_before=json.loads((direct/name/'additional_views.json').read_text())
            extra_after=json.loads((completed/name/'additional_views.json').read_text())
            if [v['image_name'] for v in extra_before['views']] != [v['image_name'] for v in extra_after['views']]:
                raise ValueError('Additional evaluation views differ')
            pair['additional_view_count']=extra_before['view_count']
            pair['direct_additional_view_iou']=extra_before['mean_visible_mask_iou']
            pair['completed_additional_view_iou']=extra_after['mean_visible_mask_iou']
            pair['additional_view_iou_delta']=extra_after['mean_visible_mask_iou']-extra_before['mean_visible_mask_iou']
        pairs.append(pair)
        original=Image.open(direct/'view_00.png').convert('RGBA')
        full=Image.open(completed/'view_00.png').convert('RGBA')
        images=[]
        for im in [original,full]:
            bg=Image.new('RGB',im.size,'white');bg.paste(im,mask=im.getchannel('A'));images.append(bg)
        for path in [direct,completed]:
            images.append(Image.open(path/name/'observed_view_comparison.jpg').crop((512,32,1024,544)))
        titles=['Observed RGB','API completion + observed','Direct mesh','Completed-input mesh']
        y=64+row*(h+label_h)
        for col,(im,title) in enumerate(zip(images,titles)):
            draw.text((col*w+8,y+8),title,font=font(18),fill='black')
            overview.paste(im.resize((w,h),Image.Resampling.LANCZOS),(col*w,y+label_h))
        draw.text((8,y+label_h+4),before['category'],font=font(18),fill='black')
        # Three matched real camera views; no perspective changes between columns.
        board=Image.new('RGB',(512*3,3*512+40),'white')
        ImageDraw.Draw(board).text((10,8),'Observed RGB | direct mesh | API-completed-input mesh',font=font(21),fill='black')
        a=Image.open(direct/name/'observed_view_comparison.jpg')
        b=Image.open(completed/name/'observed_view_comparison.jpg')
        for j in range(3):
            source_y=32+j*512;dest_y=40+j*512
            board.paste(a.crop((0,source_y,512,source_y+512)),(0,dest_y))
            board.paste(a.crop((512,source_y,1024,source_y+512)),(512,dest_y))
            board.paste(b.crop((512,source_y,1024,source_y+512)),(1024,dest_y))
        board.save(args.root/f'paired_{ident}_{args.sampler}.jpg',quality=92)
    overview.save(args.root/f'comparison_{args.sampler}.jpg',quality=94)
    report={'protocol':'same three real cameras, seed and WorldSculpt weights; opaque observed pixels preserved; API pixels alpha-composited behind antialiased observed edges',
            'scope':f'{len(args.instances)} paired objects, one seed. Input-view fit and topology proxies, not held-out or complete ground-truth 3D accuracy.',
            'sampler':args.sampler,'pairs':pairs}
    (args.root/f'paired_metrics_{args.sampler}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
