import asyncio
import json
import os
import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from preset_guard_v2 import PresetGuard, effective_prompt_logprobs
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('selfcheck_v2',Path(__file__).with_name('selfcheck-v2.py'))
selfcheck=importlib.util.module_from_spec(spec);spec.loader.exec_module(selfcheck)
check=selfcheck.check
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

class GuardTests(unittest.TestCase):
 def app(self,mode='dflash2',enabled=True,gmu='.92',actual_gmu=None):
  app=FastAPI();seen=[]
  if actual_gmu is not None:
   from types import SimpleNamespace as N
   app.state.vllm_config=N(cache_config=N(gpu_memory_utilization=actual_gmu))
  @app.get('/health')
  def health():return {'ok':True}
  @app.post('/v1/completions')
  def completion(req:CompletionRequest):
   seen.append(req.to_sampling_params(1,{}));return {'ok':True}
  @app.post('/v1/chat/completions')
  def chat(req:ChatCompletionRequest):
   seen.append(req.to_sampling_params(1,{}));return {'ok':True}
  app.add_middleware(PresetGuard,enabled=enabled,mode=mode,gmu=gmu)
  return TestClient(app),seen
 def test_non_none_including_zero_false_and_stream_never_reaches_backend(self):
  c,seen=self.app()
  for value in (0,1,20,-1,False,'1',[],{}):
   for stream in (True,False):
    with self.subTest(value=value,stream=stream),self.assertLogs('vllm.preset_guard',level='WARNING') as logs:
     r=c.post('/v1/completions',json={'model':'m','prompt':'x','prompt_logprobs':value,'stream':stream},headers={'X-Request-Id':'cpu-test-123'})
    self.assertEqual(r.status_code,400);self.assertEqual(r.json()['error']['param'],'prompt_logprobs')
    self.assertEqual(r.headers['X-Request-Id'],'cpu-test-123');self.assertIn('cpu-test-123',str(logs.output));self.assertIn('client',str(logs.output))
  self.assertEqual(seen,[]);self.assertEqual(c.get('/health').status_code,200)
 def test_missing_null_allowed_and_nospec_logs(self):
  c,seen=self.app()
  for extra in ({},{'prompt_logprobs':None}):
   self.assertEqual(c.post('/v1/completions',json=dict(model='m',prompt='x',**extra)).status_code,200)
  self.assertEqual(len(seen),2)
  c,seen=self.app('nospec',gmu='.80')
  with self.assertLogs('vllm.preset_guard',level='WARNING') as logs:
   r=c.post('/v1/completions',json=dict(model='m',prompt='x',prompt_logprobs=1))
  self.assertEqual(r.status_code,200);self.assertEqual(seen[0].prompt_logprobs,1);self.assertIn('observe',str(logs.output))
 def test_disabled_is_passthrough(self):
  c,seen=self.app(enabled=False)
  r=c.post('/v1/completions',json=dict(model='m',prompt='x',prompt_logprobs=1))
  self.assertEqual(r.status_code,200);self.assertEqual(len(seen),1)
 def test_aliases_match_real_protocol_sampling_params(self):
  for cls,path,base in [(CompletionRequest,'/v1/completions',{'prompt':'x'}),(ChatCompletionRequest,'/v1/chat/completions',{'messages':[{'role':'user','content':'x'}]})]:
   for echo in (False,True,'false','true',0,1,'on','off'):
    for extra in ({},{'logprobs':0} if cls is CompletionRequest else {'top_logprobs':0},{'prompt_logprobs':None}):
     body=dict(model='m',echo=echo,**base,**extra);req=cls.model_validate(body);actual=req.to_sampling_params(1,{}).prompt_logprobs
     calculated,_=effective_prompt_logprobs(body,path);self.assertEqual(calculated,actual,(path,body))
  c,seen=self.app()
  self.assertEqual(c.post('/v1/chat/completions',json={'model':'m','messages':[{'role':'user','content':'x'}],'echo':True}).status_code,400)
  self.assertEqual(c.post('/v1/completions',json={'model':'m','prompt':'x','echo':True,'logprobs':0}).status_code,400)
  self.assertEqual(seen,[])
 def test_environment_defaults_and_fail_closed_mode(self):
  with patch.dict(os.environ,{},clear=True):self.assertFalse(PresetGuard(None).enabled)
  with patch.dict(os.environ,{'VLLM_TEAM_PRESET_GUARD':'1','C4140_SPEC':'off','VLLM_TEAM_PRESET_GMU':'.92'},clear=True):self.assertEqual(PresetGuard(None).mode,'nospec')
  with patch.dict(os.environ,{'VLLM_TEAM_PRESET_GUARD':'1'},clear=True):
   with self.assertRaises(ValueError):PresetGuard(None)
 def test_all_modes_gmu_threshold_and_backend_spy(self):
  for mode in ('dflash2','nospec'):
   for gmu in ('.80','.879999','.88','.92','1'):
    c,seen=self.app(mode,gmu=gmu)
    r=c.post('/v1/completions',json=dict(model='m',prompt='x',prompt_logprobs=0))
    blocked=float(gmu)>=.88
    self.assertEqual(r.status_code,400 if blocked else 200,(mode,gmu))
    self.assertEqual(len(seen),0 if blocked else 1)
 def test_real_engine_config_overrides_stale_env(self):
  c,seen=self.app('nospec',gmu='.80',actual_gmu=.92)
  r=c.post('/v1/completions',json=dict(model='m',prompt='x',prompt_logprobs=1))
  self.assertEqual(r.status_code,400);self.assertEqual(r.headers['x-preset-guard-gmu'],'0.92');self.assertEqual(seen,[])
 def test_invalid_missing_gmu_fails_closed(self):
  from preset_guard_v2 import parse_gmu
  for v in (None,'nan','inf','-.1','0','1.001','oops'):
   with self.assertRaises(ValueError):parse_gmu(v)
  with patch.dict(os.environ,{'VLLM_TEAM_PRESET_GUARD':'1','C4140_SPEC':'nospec'},clear=True):
   with self.assertRaises(ValueError):PresetGuard(None)
 def test_selfcheck_real_nospec92_and_dflash92_guard(self):
  for mode in ('nospec','dflash2'):
   c,seen=self.app(mode,gmu='.92')
   def request(base,path,body=None,request_id=None,timeout=None):
    r=c.get(path) if body is None else c.post(path,json=body,headers={'X-Request-Id':request_id})
    return r.status_code,r.content,dict(r.headers)
   result=check('http://test',mode,'.92','m',request=request,pause=lambda _:None)
   self.assertEqual(result['verdict'],'PASS',result);self.assertEqual(result['http_status'],400)
   self.assertEqual(result['receipt_gmu'],'0.92');self.assertEqual(result['post_health'],[200]*3);self.assertEqual(seen,[])
 def test_selfcheck_nospec92_accidental_allow_fails(self):
  c,seen=self.app('nospec',gmu='.80')
  def request(base,path,body=None,request_id=None,timeout=None):
   r=c.get(path) if body is None else c.post(path,json=body,headers={'X-Request-Id':request_id})
   return r.status_code,r.content,dict(r.headers)
  result=check('http://test','nospec','.92','m',request=request,pause=lambda _:None)
  self.assertEqual(result['verdict'],'FAIL');self.assertEqual(result['http_status'],200)
 def test_selfcheck_http_health_failures(self):
  for mode,gmu,status,h,expected in [('dflash2','.92',400,200,'PASS'),('dflash2','.80',200,200,'PASS'),('nospec','.92',200,200,'FAIL'),('nospec','.92',400,200,'PASS'),('nospec','.80',200,200,'PASS'),('nospec','.92',500,200,'FAIL'),('nospec','.92',400,503,'FAIL')]:
   calls=[]
   def request(base,path,body=None,request_id=None,timeout=None):
    calls.append(path)
    if path=='/health':return (200 if len(calls)==1 else h),b'',{}
    self.assertEqual(len(body['prompt']),16384);self.assertEqual(body['prompt_logprobs'],1);self.assertEqual(body['max_tokens'],1)
    return status,json.dumps({'error':{'param':'prompt_logprobs','code':'team_preset_prompt_logprobs'}}).encode(),{'x-request-id':request_id,'x-preset-guard-gmu':gmu}
   self.assertEqual(check('http://unused',mode,gmu,'m',request=request,pause=lambda _:None)['verdict'],expected)
 def test_selfcheck_wrong_receipt_is_not_pass(self):
  for bad in ({'param':'other','code':'team_preset_prompt_logprobs'},{'param':'prompt_logprobs','code':400}):
   def request(base,path,body=None,request_id=None,timeout=None):
    if path=='/health':return 200,b'',{}
    return 400,json.dumps({'error':bad}).encode(),{'x-request-id':request_id,'x-preset-guard-gmu':'.92'}
   self.assertEqual(check('http://unused','nospec','.92','m',request=request,pause=lambda _:None)['verdict'],'FAIL')

class ChunkTests(unittest.IsolatedAsyncioTestCase):
 async def test_pass_body_bytes_chunks_and_malformed_unchanged(self):
  for raw in (b'{"prompt_logprobs":null,"prompt":"x"}',b'not-json',b'{"prompt":"'+b'x'*100000+b'"}'):
   seen=[];sent=[];msgs=iter([{'type':'http.request','body':raw[:7],'more_body':True},{'type':'http.request','body':raw[7:],'more_body':False}])
   async def receive():return next(msgs)
   async def send(m):sent.append(m)
   async def app(scope,receive,send):
    while True:
     m=await receive();seen.append(m['body'])
     if not m.get('more_body'):break
   scope={'type':'http','method':'POST','path':'/v1/completions','headers':[],'client':('127.0.0.1',1)}
   await PresetGuard(app,enabled=True,mode='dflash2',gmu='.92')(scope,receive,send)
   self.assertEqual(b''.join(seen),raw);self.assertEqual(len(seen),2)
 async def test_batch_and_root_path_blocked(self):
  for path,root in [('/v1/chat/completions/batch',''),('/proxy/v1/chat/completions/','/proxy')]:
   async def receive():return {'type':'http.request','body':b'{"echo":true}'}
   async def app(*_):raise AssertionError('blocked request reached backend')
   sent=[]
   async def send(m):sent.append(m)
   await PresetGuard(app,enabled=True,mode='dflash2',gmu='.92')({'type':'http','method':'POST','path':path,'root_path':root},receive,send)
   self.assertEqual(sent[0]['status'],400)
if __name__=='__main__':unittest.main(verbosity=2)
