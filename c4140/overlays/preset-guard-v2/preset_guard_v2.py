"""External ASGI middleware for 1Cat --middleware; no vLLM/core imports."""
from decimal import Decimal, InvalidOperation
import json
import logging
import os
import uuid

LOG = logging.getLogger('vllm.preset_guard')
PATHS = {'/v1/completions', '/v1/chat/completions', '/v1/chat/completions/batch', '/v1/responses'}


def effective_prompt_logprobs(body, path):
    """Mirror the two OpenAI protocol echo aliases, including chat's default 0."""
    if not isinstance(body, dict):
        return None, None
    if body.get('prompt_logprobs') is not None:
        return body['prompt_logprobs'], 'explicit'
    echo = body.get('echo')
    truthy = echo is True or type(echo) in (int, float) and echo == 1 or isinstance(echo, str) and echo.lower() in {'1','true','t','yes','y','on'}
    if truthy:
        if path.startswith('/v1/chat/completions'):
            return body.get('top_logprobs', 0), 'echo/top_logprobs'
        if path == '/v1/completions':
            return body.get('logprobs'), 'echo/logprobs'
    return None, None


THRESHOLD = Decimal('0.88')


def parse_gmu(value):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError('PRESET_GUARD requires a finite GPU memory utilization in (0,1]') from None
    if not number.is_finite() or not 0 < number <= 1:
        raise ValueError('PRESET_GUARD requires a finite GPU memory utilization in (0,1]')
    return number


def blocks_prompt_logprobs(gmu):
    return parse_gmu(gmu) >= THRESHOLD


class PresetGuard:
    def __init__(self, app, *, enabled=None, mode=None, gmu=None):
        self.app = app
        self.enabled = os.getenv('VLLM_TEAM_PRESET_GUARD') == '1' if enabled is None else enabled
        configured = mode if mode is not None else os.getenv('C4140_SPEC', '')
        self.mode = 'nospec' if configured in ('off','nospec') else configured
        if self.enabled and self.mode not in ('dflash2','nospec'):
            raise ValueError('PRESET_GUARD requires C4140_SPEC=dflash2|off when enabled')
        raw_gmu = gmu if gmu is not None else os.getenv('VLLM_TEAM_PRESET_GMU')
        self.gmu = parse_gmu(raw_gmu) if self.enabled or raw_gmu is not None else None
        LOG.warning('PRESET_GUARD_CONFIG %s', json.dumps(dict(enabled=self.enabled, mode=self.mode,
            gmu=str(self.gmu), threshold=str(THRESHOLD), block=self.enabled and blocks_prompt_logprobs(self.gmu))))

    async def __call__(self, scope, receive, send):
        path = scope.get('path','');root = scope.get('root_path','')
        if root and path.startswith(root + '/'):
            path = path[len(root):]
        path = path.rstrip('/')
        if not self.enabled or scope['type'] != 'http' or scope.get('method') != 'POST' or path not in PATHS:
            return await self.app(scope, receive, send)
        # The actual initialized engine config wins over launcher metadata. No tensor access.
        state = getattr(scope.get('app'), 'state', None)
        cfg = getattr(state, 'vllm_config', None)
        cache_cfg = getattr(cfg, 'cache_config', None)
        actual_gmu = getattr(cache_cfg, 'gpu_memory_utilization', None)
        gmu = parse_gmu(actual_gmu) if actual_gmu is not None else self.gmu
        reject = blocks_prompt_logprobs(gmu)
        messages=[]
        while True:
            message=await receive()
            if message['type']=='http.disconnect':return
            messages.append(message)
            if not message.get('more_body',False):break
        try:
            body=json.loads(b''.join(m.get('body',b'') for m in messages))
        except (ValueError, UnicodeError):
            body=None  # Existing API owns JSON/type validation; bytes remain unchanged.
        value, source=effective_prompt_logprobs(body,path)
        if value is not None:
            headers=list(scope.get('headers',[]))
            supplied=next((v.decode('latin1') for k,v in headers if k.lower()==b'x-request-id'),None)
            request_id=supplied if supplied is not None else 'preset-'+uuid.uuid4().hex
            if supplied is None:
                scope=dict(scope, headers=headers+[(b'x-request-id',request_id.encode('ascii'))])
            client=scope.get('client')
            # Structured log escapes CR/LF; never log prompt text, bearer token or forwarded IP.
            LOG.warning('PRESET_GUARD %s',json.dumps(dict(mode=self.mode,gmu=str(gmu),gmu_source='engine-config' if actual_gmu is not None else 'launcher',
                threshold=str(THRESHOLD),action='reject' if reject else 'observe',
                request_id=request_id,client=list(client) if client else None,path=path,source=source,
                prompt_logprobs=value if isinstance(value,(int,float,str,bool)) else type(value).__name__),ensure_ascii=True))
            if reject:
                payload=json.dumps({'error':{'message':'prompt_logprobs (including echo aliases) is disabled in team presets with gpu_memory_utilization >= 0.88.',
                    'type':'BadRequestError','param':'prompt_logprobs','code':'team_preset_prompt_logprobs'}}).encode()
                await send({'type':'http.response.start','status':400,'headers':[(b'content-type',b'application/json'),
                    (b'content-length',str(len(payload)).encode()),(b'x-request-id',request_id.encode('latin1')),
                    (b'x-preset-guard-gmu',str(gmu).encode('ascii'))]})
                await send({'type':'http.response.body','body':payload});return
        pending=iter(messages)
        async def replay():
            try:return next(pending)
            except StopIteration:return await receive()
        await self.app(scope,replay,send)
