"""Lit photo colors: linear USD/GLB colors and explicit dielectric PBR.

Input ColorVisuals contain sRGB bytes. Linear uint16 GLB colors preserve dark
gradients. These colors include photographed illumination, not intrinsic albedo.
"""
import numpy as np
from trimesh.exchange.gltf import export_glb


def srgb_to_linear(colors):
    colors = np.asarray(colors)
    return np.where(colors <= .04045, colors / 12.92,
                    ((colors + .055) / 1.055) ** 2.4)


def _linear_colors(buffers, tree):
    keys = list(buffers)
    accessors = list(tree['accessors'].values())
    indices = {p['attributes']['COLOR_0'] for m in tree['meshes']
               for p in m['primitives'] if 'COLOR_0' in p['attributes']}
    converted = set()
    for index in indices:
        accessor = accessors[index]
        assert accessor['componentType'] == 5121 and accessor['type'] == 'VEC4'
        assert accessor.get('byteOffset', 0) == 0
        key = keys[accessor['bufferView']]
        if key not in converted:
            rgba = np.frombuffer(buffers[key], np.uint8).reshape(-1, 4) / 255.
            rgba[:, :3] = srgb_to_linear(rgba[:, :3])
            values = np.rint(rgba * 65535).astype('<u2')
            buffers[key] = values.tobytes()
            converted.add(key)
        else:
            values = np.frombuffer(buffers[key], '<u2').reshape(-1, 4)
        accessor.update(componentType=5123, normalized=True,
                        min=values.min(0).tolist(), max=values.max(0).tolist())


def _dielectric_material(tree):
    materials = tree.setdefault('materials', [])
    index = len(materials)
    materials.append({
        'name': 'ObservedVertexColor',
        'pbrMetallicRoughness': {
            'baseColorFactor': [1., 1., 1., 1.],
            'metallicFactor': 0., 'roughnessFactor': .85,
        },
        'doubleSided': True,
        'extras': {'colorSource': 'Observed photos including baked illumination; not intrinsic albedo'},
    })
    for mesh in tree['meshes']:
        for primitive in mesh['primitives']:
            assert 'COLOR_0' in primitive['attributes']
            primitive['material'] = index


def export_observed_glb(mesh_or_scene, path):
    path.write_bytes(export_glb(mesh_or_scene, include_normals=True,
                               buffer_postprocessor=_linear_colors,
                               tree_postprocessor=_dielectric_material))


# Reproducible viewing preset; not an estimate of the original room lights.
VIEW_DOME_INTENSITY = 4800.
VIEW_RENDER_SETTINGS = {
    'rtx:rendermode': 'RaytracedLighting',
    'rtx:post:histogram:enabled': False,
    'rtx:post:tonemap:op': 2,
    'rtx:post:tonemap:filmIso': 100.,
    'rtx:post:tonemap:exposureTime': .02,
    'rtx:post:tonemap:fNumber': 5.,
    'rtx:post:tonemap:cm2Factor': 1.,
    'rtx:post:tonemap:enableSrgbToGamma': True,
}
