import onnx, numpy as np
from onnx import shape_inference, numpy_helper
p = r"C:\Microchip\hailo-stereo\artifacts\stereonet_pretrained\stereonet.onnx"
m = shape_inference.infer_shapes(onnx.load(p))
g = m.graph
vi = {v.name: v for v in list(g.value_info)+list(g.input)+list(g.output)}
init = {t.name: numpy_helper.to_array(t) for t in g.initializer}

def s(n):
    if n in init: return f"CONST{list(init[n].shape)}"
    if n in vi:
        d=[(x.dim_value if x.HasField('dim_value') else '?') for x in vi[n].type.tensor_type.shape.dim]
        return str(d)
    return "?"

def attrs(node):
    out=[]
    for a in node.attribute:
        if a.name in ("kernel_shape","strides","pads","dilations","axes","starts","ends","perm","group","axis"):
            v = list(a.ints) if a.ints else (a.i if a.type==2 else a.f)
            out.append(f"{a.name}={v}")
        elif a.name=="alpha": out.append(f"alpha={a.f}")
        elif a.name=="mode": out.append(f"mode={a.s.decode()}")
    return " ".join(out)

for i,node in enumerate(g.node):
    ins = " ".join(f"{x}{s(x)}" for x in node.input)
    outs = " ".join(f"{x}{s(x)}" for x in node.output)
    print(f"[{i:3d}] {node.op_type:12s} {outs}   <=  {ins}   {attrs(node)}")
