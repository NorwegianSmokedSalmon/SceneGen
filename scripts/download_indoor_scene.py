#!/usr/bin/env python3
"""Download only selected Mip-NeRF 360 indoor images + COLMAP from the official ZIP.

HTTP ranges avoid downloading unrelated scenes and full-resolution images.
The original COLMAP model is preserved in source_sparse/, and the working model
is scaled to the actual downloaded image dimensions (data_factor=1).
"""
import argparse
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import requests

URL = 'https://storage.googleapis.com/gresearch/refraw360/360_v2.zip'


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        probe = self.session.get(url, headers={'Range': 'bytes=0-0'}, params={'scene_gen_range':'0-0'}, timeout=60)
        probe.raise_for_status()
        if probe.status_code != 206:
            raise RuntimeError('Server must support byte ranges; refusing full ZIP download.')
        self.size = int(probe.headers['Content-Range'].split('/')[-1])
        self.etag = probe.headers.get('ETag')
        self.position = 0
        self.block_size = 1024 * 1024
        self.cache = OrderedDict()
        self.downloaded = len(probe.content)

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if position < 0: raise ValueError('Negative seek')
        self.position = position
        return position

    def block(self, index):
        if index not in self.cache:
            start = index * self.block_size
            end = min(start + self.block_size, self.size) - 1
            headers = {'Range': f'bytes={start}-{end}'}
            if self.etag: headers['If-Match'] = self.etag
            for attempt in range(3):
                try:
                    r = self.session.get(self.url, headers=headers, params={'scene_gen_range':f'{start}-{end}'}, timeout=90)
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}' or len(r.content) != end-start+1:
                        raise RuntimeError('Unexpected byte range response')
                    break
                except (requests.RequestException, RuntimeError):
                    if attempt == 2: raise
            self.cache[index] = r.content
            self.downloaded += len(r.content)
            if len(self.cache) > 8: self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def read(self, size=-1):
        if size < 0: size = self.size-self.position
        end = min(self.position+size, self.size)
        chunks = []
        while self.position < end:
            index, offset = divmod(self.position, self.block_size)
            chunk = self.block(index)[offset:offset+min(end-self.position, self.block_size-offset)]
            chunks.append(chunk)
            self.position += len(chunk)
        return b''.join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', nargs='+', choices=['bonsai','room','kitchen','counter'], default=['bonsai','room'])
    parser.add_argument('--output', type=Path, default=Path('data/mipnerf360_indoor'))
    parser.add_argument('--factor', type=int, choices=[2,4,8], default=4)
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    remote = RangeFile(URL)
    with zipfile.ZipFile(remote) as archive:
        members = archive.infolist()
        for scene in args.scenes:
            image_prefix = f'{scene}/images_{args.factor}/'
            selected = [i for i in members if not i.is_dir() and
                        (i.filename.startswith(image_prefix) or i.filename.startswith(f'{scene}/sparse/0/'))]
            print(scene, 'files:',len(selected),'uncompressed MB:',round(sum(i.file_size for i in selected)/1e6), flush=True)
            if args.list_only:
                print([i.filename for i in selected[:3]])
                continue
            if not selected: raise RuntimeError(f'Missing scene {scene}')
            root = args.output / scene
            records = []
            for index, info in enumerate(selected):
                relative = PurePosixPath(info.filename).relative_to(scene)
                if '..' in relative.parts: raise RuntimeError('Unexpected archive path')
                dest = root / ('images' if relative.parts[0].startswith('images_') else 'source_sparse') / Path(*relative.parts[1:])
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    data = dest.read_bytes()
                    import zlib
                    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
                        raise RuntimeError(f'Existing file differs: {dest}')
                else:
                    data = archive.read(info)  # zipfile verifies CRC.
                    partial = dest.with_name(dest.name+'.part')
                    partial.write_bytes(data)
                    partial.replace(dest)
                records.append({'archive_path':info.filename,'path':str(dest.relative_to(root)), 'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
                if index % 40 == 0: print(scene,index+1,'/',len(selected),flush=True)
            prepare_colmap(root)
            (root/'download_manifest.json').write_text(json.dumps(dict(source=URL,etag=remote.etag,scene=scene,downsample_factor=args.factor,files=records),indent=2)+'\n')
            print('Ready:',root,flush=True)
    print('HTTP bytes transferred:',remote.downloaded,flush=True)


def prepare_colmap(root):
    import numpy as np
    import pycolmap
    from PIL import Image
    rec = pycolmap.Reconstruction(root/'source_sparse/0')
    paths = sorted((root/'images').iterdir())
    by_stem = {p.stem.lower():p for p in paths}
    scales = {}
    for cam_id, camera in rec.cameras.items():
        image = next(im for im in rec.images.values() if im.camera_id == cam_id)
        path = by_stem[Path(image.name).stem.lower()]
        with Image.open(path) as im: width,height = im.size
        scales[cam_id] = np.array([width/camera.width,height/camera.height])
        camera.rescale(width,height)
    for image in rec.images.values():
        image.name = by_stem[Path(image.name).stem.lower()].name
        for point in image.points2D:
            point.xy = point.xy * scales[image.camera_id]
    dest = root/'sparse/0'
    dest.mkdir(parents=True,exist_ok=True)
    rec.write(dest)
    print('COLMAP:',rec.num_reg_images(),'registered images;',rec.num_points3D(),'points',flush=True)


if __name__ == '__main__':
    main()
