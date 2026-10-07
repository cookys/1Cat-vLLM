"""Host-side exhaustive check: new integer scale decode == old float path.
Old (fp8_kv_utils.cuh pre-change): fp8_e4m3fn_to_float -> __float2half_rn,
  NaN: quiet_nan_f 0x7fffffff -> half 0x7fff.
New: mirrors nvfp4_data16_to_half32 bit-for-bit (PRMT pair, shifts, arithmetic
  NaN mask, HMUL2 by 0x5c00 = 256, PRMT broadcast).
Run with any python that has numpy (e.g. /data/venvs/1cat-p070/bin/python)."""
import numpy as np
M32=0xffffffff
def f16(bits): return np.array([bits],dtype=np.uint16).view(np.float16)[0]
def old(raw):
    s=raw>>7; e=(raw>>3)&15; m=raw&7; sg=-1 if s else 1
    if raw&0x7f==0: v=-0.0 if s else 0.0
    elif e==0: v=sg*m*2.0**-9
    elif e==15 and m==7: return 0x7fff
    else: v=sg*(1+m*0.125)*2.0**(e-7)
    return int(np.float16(v).view(np.uint16))
def hmul256(hb):  # HMUL2 lane; NaN in -> canonical NaN 0x7fff (PTX mul.f16)
    h=f16(hb)
    if np.isnan(h): return 0x7fff
    return int(np.array([h*np.float16(256)]).astype(np.float16).view(np.uint16)[0])
def new_pair(scales2):
    pair=(scales2&0xff)|(((scales2>>8)&0xff)<<16)                 # PRMT 0x4140
    raw_h=(((pair<<7)&0x3f803f80)|((pair<<8)&0x80008000))&M32
    y=(((pair&0x007f007f)+0x00010001)&M32)&0x00800080
    nanm=((y<<8)-(y>>7))&M32
    raw_h=((raw_h&~((y<<8)&M32)&M32)|nanm)&M32
    m=hmul256(raw_h&0xffff)|(hmul256(raw_h>>16)<<16)
    s0=(m&0xffff)*0x10001; s1=(m>>16)*0x10001                     # PRMT 0x1010/0x3232
    return s0,s1
def test_scale_decode_bit_identical():
    bad=0
    for sc in range(65536):
        s0,s1=new_pair(sc)
        e0=old(sc&0xff); e1=old(sc>>8)
        if (s0,s1)!=(e0*0x10001,e1*0x10001): bad+=1
    # upper 16 bits of scales2 must not matter
    for hi in (0xffff,0x1234,0x7f7f):
        for lo in range(0,65536,251): assert new_pair(lo|(hi<<16))==new_pair(lo)
    # exactness: x256 product is exact (no rounding) for every non-NaN code
    for r in range(256):
        if r&0x7f==0x7f: continue
        hb=((r<<7)&0x3f80)|((r<<8)&0x8000)
        assert float(f16(hb))*256.0==float(f16(old(r)))
    print("mismatching (b0,b1) pairs: %d of 65536 (covers all 256 single bytes in both lanes)"%bad)
    assert bad==0
    print("OK bit-identical")


if __name__ == '__main__':
    test_scale_decode_bit_identical()
