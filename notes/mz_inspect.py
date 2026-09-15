import onnx, numpy as np, collections, os
from onnx import numpy_helper

p = r"C:\Microchip\hailo-stereo\artifacts\stereonet_pretrained\stereonet.onnx"
m = onnx.load(p)
g = m.graph
print("=== META ===")
print("ir_version:", m.ir_version, "producer:", m.producer_name, m.producer_version)
for o in m.opset_import: print("  opset:", o.domain or "ai.onnx", o.version)
print("file MB: %.2f" % (os.path.getsize(p)/1e6))

def shp(vi):
    t = vi.type.tensor_type
    d = [(x.dim_value if x.HasField('dim_value') else (x.dim_param or '?')) for x in t.shape.dim]
    return f"{vi.name} {onnx.TensorProto.DataType.Name(t.elem_type)} {d}"

print("\n=== INPUTS ===")
for i in g.input: print(" ", shp(i))
print("=== OUTPUTS ===")
for o in g.output: print(" ", shp(o))

print("\n=== NODES: %d ===" % len(g.node))
c = collections.Counter(n.op_type for n in g.node)
for k,v in c.most_common(): print(f"  {k:22s} {v}")

print("\n=== INITIALIZERS ===")
inits = list(g.initializer)
tot = 0; rows=[]
for t in inits:
    a = numpy_helper.to_array(t)
    b = a.nbytes; tot += b
    rows.append((b, a.size, t.name, str(a.dtype), tuple(a.shape)))
print(f"count={len(inits)} total_bytes={tot/1e6:.2f} MB  total_params={sum(r[1] for r in rows)/1e6:.3f} M")
rows.sort(reverse=True)
print("\n-- top 20 largest initializers --")
for b,sz,n,dt,s in rows[:20]:
    print(f"  {b/1e6:9.3f} MB  {sz:>12,}  {dt:8s} {str(s):28s} {n}")
