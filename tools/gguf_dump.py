#!/usr/bin/env python3
"""Stdlib-only GGUF inspector: metadata + tensor table (type, shape, bytes) across split files."""
import struct, sys, json, collections

GGML_TYPES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",10:"Q2_K",11:"Q3_K",12:"Q4_K",
  13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",
  23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",29:"IQ1_M",30:"BF16",34:"TQ1_0",35:"TQ2_0",39:"MXFP4"}
# (block elems, block bytes)
BLK = {"F32":(1,4),"F16":(1,2),"BF16":(1,2),"Q8_0":(32,34),"Q4_0":(32,18),"Q4_1":(32,20),"Q5_0":(32,22),"Q5_1":(32,24),
  "Q2_K":(256,84),"Q3_K":(256,110),"Q4_K":(256,144),"Q5_K":(256,176),"Q6_K":(256,210),"Q8_K":(256,292),
  "IQ2_XXS":(256,66),"IQ2_XS":(256,74),"IQ3_XXS":(256,98),"IQ1_S":(256,50),"IQ4_NL":(32,18),"IQ3_S":(256,110),
  "IQ2_S":(256,82),"IQ4_XS":(256,136),"IQ1_M":(256,56),"I32":(1,4),"MXFP4":(32,17),"TQ1_0":(256,54),"TQ2_0":(256,66)}

def rd(f, fmt):
    s = struct.calcsize("<"+fmt); return struct.unpack("<"+fmt, f.read(s))
def rstr(f):
    (n,) = rd(f, "Q"); return f.read(n).decode("utf-8", "replace")
def rval(f, t):
    if t == 0: return rd(f,"B")[0]
    if t == 1: return rd(f,"b")[0]
    if t == 2: return rd(f,"H")[0]
    if t == 3: return rd(f,"h")[0]
    if t == 4: return rd(f,"I")[0]
    if t == 5: return rd(f,"i")[0]
    if t == 6: return rd(f,"f")[0]
    if t == 7: return bool(rd(f,"B")[0])
    if t == 8: return rstr(f)
    if t == 9:
        (at, n) = rd(f, "IQ")
        return [rval(f, at) for _ in range(n)]
    if t == 10: return rd(f,"Q")[0]
    if t == 11: return rd(f,"q")[0]
    if t == 12: return rd(f,"d")[0]
    raise ValueError(t)

def parse(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        (ver, nt, nkv) = rd(f, "IQQ")
        kv = {}
        for _ in range(nkv):
            k = rstr(f); (t,) = rd(f, "I"); kv[k] = rval(f, t)
        tens = []
        for _ in range(nt):
            name = rstr(f); (nd,) = rd(f, "I"); dims = rd(f, "Q"*nd); (tt, off) = rd(f, "IQ")
            tens.append((name, dims, GGML_TYPES.get(tt, str(tt)), off))
        align = kv.get("general.alignment", 32)
        data_off = (f.tell() + align - 1) // align * align
    return kv, tens, data_off

def nbytes(dims, t):
    n = 1
    for d in dims: n *= d
    be, bb = BLK.get(t, (1, 0))
    return n // be * bb

if __name__ == "__main__":
    allkv, alltens = {}, []
    for p in sys.argv[1:]:
        kv, tens, doff = parse(p)
        for k, v in kv.items():
            if k not in allkv: allkv[k] = v
        for (n, d, t, o) in tens: alltens.append((n, d, t, o, p.split("/")[-1], doff))
    for k, v in allkv.items():
        if isinstance(v, list) and len(v) > 16: v = f"[{len(v)} items] {v[:8]}..."
        print(f"KV {k} = {v}")
    by_type = collections.Counter(); by_cls = collections.Counter()
    for (n, d, t, o, fn, doff) in alltens:
        b = nbytes(d, t); by_type[t] += b
        cls = "exps" if "_exps" in n else ("shexp" if "shexp" in n else ("embd/out" if ("token_embd" in n or n.startswith("output")) else "dense"))
        by_cls[cls] += b
    for (n, d, t, o, fn, doff) in alltens:
        if n.startswith("blk.") and not (n.startswith("blk.0.") or n.startswith("blk.1.") or n.startswith("blk.3.")): continue
        print(f"T {n:50s} {t:8s} {str(list(d)):28s} {nbytes(d,t)/2**20:10.2f} MiB  {fn[-20:]}")
    print("BYTES by type (GiB):", {k: round(v/2**30, 2) for k, v in by_type.most_common()})
    print("BYTES by class (GiB):", {k: round(v/2**30, 2) for k, v in by_cls.most_common()})
    print("n_tensors", len(alltens))
