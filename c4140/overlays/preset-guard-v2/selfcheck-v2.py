#!/usr/bin/env python3
"""Post-boot client probe. Plan-only by default; never starts/stops a server."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import urllib.error
import uuid
from preset_guard_v2 import blocks_prompt_logprobs, parse_gmu


def http(base,path,body=None,request_id=None,timeout=120):
    headers={'Content-Type':'application/json'}
    if request_id:headers['X-Request-Id']=request_id
    req=urllib.request.Request(base+path,headers=headers,data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:return r.status,r.read(),dict(r.headers)
    except urllib.error.HTTPError as e:return e.code,e.read(),dict(e.headers)


def check(base,mode,gmu,model=None,request=http,pause=time.sleep):
    gmu=parse_gmu(gmu)
    must_block=blocks_prompt_logprobs(gmu)
    rid='preset-selfcheck-'+uuid.uuid4().hex
    result=dict(request_id=rid,preset=mode,gmu=str(gmu),must_block=must_block,ctx_tokens=16384,prompt_logprobs=1,verdict='FAIL')
    try:
        code,_,_=request(base,'/health',timeout=5)
        if code!=200:raise RuntimeError('pre-health:'+str(code))
        if model is None:
            code,raw,_=request(base,'/v1/models',timeout=5)
            if code!=200:raise RuntimeError('models:'+str(code))
            model=json.loads(raw)['data'][0]['id']
        body=dict(model=model,prompt=[1000]*16384,prompt_logprobs=1,max_tokens=1,n=1,stream=False,temperature=0,seed=0)
        code,raw,headers=request(base,'/v1/completions',body,rid)
        result.update(http_status=code,response=raw.decode(errors='replace')[:4096],response_request_id=headers.get('x-request-id',headers.get('X-Request-Id')))
        # Survival required after 400 too. Do not retry away a transient death/503.
        health=[]
        for _ in range(3):
            h,_,_=request(base,'/health',timeout=5);health.append(h);pause(1)
        result['post_health']=health
        result['survival_ok']=code in (400,200) and all(h==200 for h in health)
        result['policy_ok']=code==400 if must_block else code in (400,200)
        # Any high-gmu mode requires the guard receipt, not unrelated validation.
        if must_block:
            error=json.loads(raw).get('error',{})
            result['policy_ok'] &= error.get('param')=='prompt_logprobs' and error.get('code')=='team_preset_prompt_logprobs' and result['response_request_id']==rid
            receipt_gmu=headers.get('x-preset-guard-gmu',headers.get('X-Preset-Guard-Gmu'))
            result['receipt_gmu']=receipt_gmu
            result['policy_ok'] &= parse_gmu(receipt_gmu)==gmu
        if result['survival_ok'] and result['policy_ok']:result['verdict']='PASS'
    except Exception as e:result['error']=str(e)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--base',default='http://127.0.0.1:8021')
    p.add_argument('--gmu',required=True,type=parse_gmu);p.add_argument('--preset',choices=['dflash2','nospec'],required=True);p.add_argument('--model');p.add_argument('--out',type=Path)
    a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(plan_only=True,base=a.base,preset=a.preset,gmu=str(a.gmu),must_block=blocks_prompt_logprobs(a.gmu),prompt_tokens=16384,prompt_logprobs=1,max_tokens=1,post_health_checks=3)));return 0
    if not a.out:p.error('--execute requires --out (outside sealed package)')
    if a.out.resolve().is_relative_to(Path(__file__).resolve().parent):p.error('--out must be outside package')
    if a.out.exists():p.error('--out exists')
    result=check(a.base,a.preset,a.gmu,a.model);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('PRESET_GUARD_CHECK='+result['verdict'],flush=True);return 0 if result['verdict']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
