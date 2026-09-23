#!/usr/bin/env python3
"""Call Qwen's public demo or ModelScope free image editing; never use a paid fallback."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen-Image-Edit-2511'
BASE = 'https://api-inference.modelscope.cn/v1'


def token_from_env():
    keys = ('MODELSCOPE_API_TOKEN','MODELSCOPE_ACCESS_TOKEN','MODELSCOPE_TOKEN','MODELSCOPE_SDK_TOKEN')
    for key in keys:
        if os.environ.get(key): return os.environ[key]
    env = ROOT/'.env'
    if env.exists():
        for line in env.read_text().splitlines():
            key, sep, value = line.strip().removeprefix('export ').partition('=')
            if sep and key.strip() in keys:
                value = value.strip()
                if value[:1] in ('"', "'") and value[-1:] == value[:1]: value=value[1:-1]
                if value: return value
    return None


def save_record(path, record):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    temp.replace(path)


def check(response):
    if response.status_code >= 400:
        details = []
        data = {}
        try:
            data = response.json()
            error = data.get('errors', data.get('error', data))
            if isinstance(error, str):
                details.append(error)
            elif isinstance(error, dict):
                for key in ('code', 'message', 'Code', 'Message', 'request_id'):
                    value = error.get(key)
                    if isinstance(value, (str, int)):
                        details.append(f'{key}={value}')
        except (ValueError, AttributeError):
            pass
        if isinstance(data, dict) and data.get('request_id'):
            details.append('request_id=' + str(data['request_id']))
        message = '; '.join(details)
        token = token_from_env()
        if token:
            message = message.replace(token, '[REDACTED]')
        import re
        message = re.sub(r'ms-[A-Za-z0-9-]+', '[REDACTED]', message)
        message = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '[IMAGE DATA]', message)
        raise RuntimeError(f'HTTP {response.status_code}: {message[:1000] or "free API request failed"}; no paid fallback')
    return response.json()


def modelscope(args, record, record_path):
    token = token_from_env()
    if not token: raise RuntimeError('Missing MODELSCOPE_API_TOKEN in project .env or environment')
    headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'}
    task_id=record.get('task_id')
    if not task_id:
        image='data:image/png;base64,'+base64.b64encode(args.image.read_bytes()).decode()
        payload={'model':MODEL,'prompt':args.prompt.read_text().strip(),'image_url':[image],
                 'seed':args.seed,'steps':args.steps,'guidance':args.guidance}
        if args.provider_defaults:
            payload.pop('steps');payload.pop('guidance')
        if args.size:
            payload['size']=args.size
        record['submitted_parameters']={k:v for k,v in payload.items() if k not in ('image_url','prompt')}
        response=requests.post(BASE+'/images/generations',headers={**headers,'X-ModelScope-Async-Mode':'true'},
                               json=payload,timeout=120)
        task_id=check(response)['task_id'];record['task_id']=task_id
        record['status']='submitted';record['submitted_utc']=datetime.now(timezone.utc).isoformat();save_record(record_path,record)
    deadline=time.monotonic()+args.timeout
    while time.monotonic()<deadline:
        data=check(requests.get(BASE+'/tasks/'+task_id,headers={**headers,'X-ModelScope-Task-Type':'image_generation'},timeout=60))
        status=data.get('task_status');print('ModelScope',status,flush=True)
        if status=='SUCCEED':
            response=requests.get(data['output_images'][0],timeout=120);response.raise_for_status()
            args.output.write_bytes(response.content);return
        if status in ('FAILED','CANCELED','CANCELLED'): raise RuntimeError('ModelScope task '+status)
        time.sleep(5)
    raise TimeoutError('Polling timed out; rerun to resume this task without submitting another')


def huggingface(args, record, record_path):
    from gradio_client import Client, handle_file
    from concurrent.futures import TimeoutError as FutureTimeout
    client=Client(args.space,verbose=False,download_files=str(args.output.parent/'downloads'))
    metadata=check(requests.get('https://huggingface.co/api/spaces/'+args.space,timeout=30))
    record['space_commit']=metadata.get('sha');save_record(record_path,record)
    job=client.submit(images=[{'image':handle_file(str(args.image))}],prompt=args.prompt.read_text().strip(),
                      seed=args.seed,randomize_seed=False,true_guidance_scale=args.guidance,num_inference_steps=args.steps,
                      height=1024,width=1536,rewrite_prompt=False,api_name='/infer')
    deadline=time.monotonic()+args.timeout
    while time.monotonic()<deadline:
        try:
            result=job.result(timeout=20)
            gallery,seed=result[:2]
            record['returned_seed']=seed
            item=gallery[0]
            if isinstance(item,dict): item=item['image']
            if isinstance(item,dict): item=item['path']
            if isinstance(item,(tuple,list)): item=item[0]
            shutil.copyfile(item,args.output)
            return
        except FutureTimeout:
            print('Hugging Face',job.status().code,flush=True)
    job.cancel();raise TimeoutError('Public demo timed out; cancelled the queued job')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider',choices=('huggingface','modelscope'),default='huggingface')
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--prompt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--space',default=MODEL)
    p.add_argument('--steps',type=int,default=40)
    p.add_argument('--guidance',type=float,default=4.)
    p.add_argument('--size',help='Explicit output dimensions, e.g. 1536x1024')
    p.add_argument('--provider-defaults',action='store_true',help='Keep the service default sampling settings')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--timeout',type=int,default=600)
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    record_path=args.output.with_suffix('.request.json')
    fingerprint_data=args.image.read_bytes()+args.prompt.read_bytes()+str((args.seed,args.provider,args.space,args.steps,args.guidance)).encode()
    if args.size: fingerprint_data+=args.size.encode()
    if args.provider_defaults: fingerprint_data+=b'provider_defaults'
    fingerprint=hashlib.sha256(fingerprint_data).hexdigest()
    record={'provider':args.provider,'model':MODEL,'seed_requested':args.seed,'fingerprint':fingerprint,
            'input':str(args.image),'output':str(args.output),'started_utc':datetime.now(timezone.utc).isoformat(),
            'status':'preparing','paid_fallback':False,'space':args.space,'steps':None if args.provider_defaults else args.steps,'guidance':None if args.provider_defaults else args.guidance}
    if record_path.exists():
        previous=json.loads(record_path.read_text())
        if previous['fingerprint']!=fingerprint: raise SystemExit('Request differs; choose a new output filename')
        record=previous
        if record.get('status')=='failed':
            record.setdefault('previous_attempts',[]).append({k:record[k] for k in ('error','error_type','started_utc','finished_utc') if k in record})
            for k in ('error','error_type','finished_utc'): record.pop(k,None)
            record['started_utc']=datetime.now(timezone.utc).isoformat()
        if record.get('status')=='succeeded' and args.output.exists(): print('Using recorded successful output');return
    save_record(record_path,record)
    try:
        globals()[args.provider](args,record,record_path)
        from PIL import Image
        with Image.open(args.output) as im:
            record['output_size']=list(im.size);im.verify()
        record['output_sha256']=hashlib.sha256(args.output.read_bytes()).hexdigest()
        record['status']='succeeded'
        record.pop('error',None);record.pop('error_type',None)
    except Exception as exc:
        record['status']='failed';record['error_type']=type(exc).__name__
        message=str(exc)
        token=token_from_env()
        if token: message=message.replace(token,'[REDACTED]')
        record['error']=message[:1200]
        print('API failed:',record['error'],flush=True)
    record['finished_utc']=datetime.now(timezone.utc).isoformat();save_record(record_path,record)
    if record['status']!='succeeded': raise SystemExit(1)
    print('Saved',args.output,record['output_size'])

if __name__=='__main__': main()
