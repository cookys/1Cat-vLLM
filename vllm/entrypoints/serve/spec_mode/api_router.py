# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""plan 075 S1: runtime DFlash2 <-> nospec soft switch.

``POST /spec_mode?mode=dflash2|nospec`` flips the scheduler's spec mode for
requests added from then on (K is fixed per request at add time). ``GET
/spec_mode`` reports the mode and how many running requests use each K.

Unlike the sleep / rpc routers this one is mounted unconditionally: the serving
launch line must not need an extra environment variable to be switchable.
"""

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

SPEC_MODES = ("dflash2", "nospec")

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/spec_mode")
async def set_spec_mode(raw_request: Request):
    mode = raw_request.query_params.get("mode")
    if mode not in SPEC_MODES:
        return JSONResponse(
            status_code=400,
            content={"error": f"mode must be one of {list(SPEC_MODES)}, got {mode!r}"},
        )
    try:
        result = await engine_client(raw_request).set_spec_mode(mode)
    except (ValueError, RuntimeError, NotImplementedError) as e:
        return JSONResponse(status_code=409, content={"error": str(e)})
    if result["changed"]:
        logger.info("spec mode %s -> %s", result["prev"], result["mode"])
    return JSONResponse(content=result)


@router.get("/spec_mode")
async def get_spec_mode(raw_request: Request):
    try:
        result = await engine_client(raw_request).get_spec_mode()
    except (ValueError, RuntimeError, NotImplementedError) as e:
        return JSONResponse(status_code=409, content={"error": str(e)})
    return JSONResponse(content=result)


def attach_router(app: FastAPI):
    app.include_router(router)
