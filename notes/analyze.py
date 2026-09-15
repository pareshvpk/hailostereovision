import onnx, numpy as np
from onnx import shape_inference, numpy_helper
p = r"C:\Microchip\hailo-stereo\artifacts\stereonet_pretrained\stereonet.onnx"
m = shape_inference.infer_shapes(onnx.load(p)); g = m.graph
vi = {v.name: v for v in list(g.value_info)+list(g.input)+list(g.output)}
init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
def sh(n):
    if n in init: return tuple(init[n].shape)
    if n in vi: return tuple(x.dim_value for x in vi[n].type.tensor_type.shape.dim)
    return None

print("="*70); print("CONSTANT INSPECTION"); print("="*70)
t = init['/Tile_output_0']
print("/Tile_output_0 shape", t.shape, "dtype", t.dtype, "bytes %.2f MB" % (t.nbytes/1e6))
print("  unique values:", np.unique(t))
print("  per-channel unique (c=0..11):", [np.unique(t[0,c]).tolist() for c in range(t.shape[1])])
print("  -> is it a constant ramp per channel?", all(len(np.unique(t[0,c]))==1 for c in range(12)))

print("\n-- shift pad constants (cost-volume) --")
for k in sorted(init):
    if k.startswith('/Constant') and init[k].ndim==4:
        a=init[k]; print(f"  {k:28s} {str(a.shape):20s} unique={np.unique(a)[:4]} allzero={np.all(a==0)}")
print("\n-- slice params --")
for k in ['/Constant_1_output_0','/Constant_2_output_0','/Constant_3_output_0','/Constant_4_output_0','/Concat_12_output_0']:
    if k in init: print(f"  {k:28s} = {init[k]}")

print("\n"+"="*70); print("MAC / PARAM BREAKDOWN"); print("="*70)
groups = {}
tot_mac=0; tot_lp=0
for n in g.node:
    if n.op_type!='Conv': continue
    w = init[n.input[1]]; o = sh(n.output[0])
    spatial = int(np.prod(o[2:]))
    mac = spatial * int(np.prod(w.shape))/w.shape[0] * o[1]
    name=n.name
    grp = ('feature' if '/downsampling/' in name or '/res/' in name else
           'cost_filter_3D' if 'cost_volume_filter' in name else
           'refine' if '/refine/' in name else 'other')
    d = groups.setdefault(grp, [0,0,0])
    d[0]+=mac; d[1]+=int(np.prod(w.shape))+w.shape[0]; d[2]+=1
    tot_mac+=mac
print(f"{'group':18s} {'convs':>6s} {'GMAC':>10s} {'GOPS':>10s} {'%OPS':>7s} {'weights':>12s}")
for k,(mac,par,cnt) in sorted(groups.items(), key=lambda x:-x[1][0]):
    print(f"{k:18s} {cnt:6d} {mac/1e9:10.2f} {2*mac/1e9:10.2f} {100*mac/tot_mac:6.1f}% {par:12,}")
print(f"{'TOTAL':18s} {sum(v[2] for v in groups.values()):6d} {tot_mac/1e9:10.2f} {2*tot_mac/1e9:10.2f} {100.0:6.1f}% {sum(v[1] for v in groups.values()):12,}")

print("\n-- learned params vs shipped initializers --")
learned = sum(v.size for k,v in init.items() if not k.startswith('/'))
consts  = sum(v.size for k,v in init.items() if k.startswith('/'))
print(f"  learned weights/biases : {learned:>12,}  ({learned/1e6:.3f} M)")
print(f"  baked graph constants  : {consts:>12,}  ({consts/1e6:.3f} M)")
print(f"  doc claims             : 623,100,000  (623.1 M)")

print("\n-- weight sharing between the two towers --")
used = {}
for n in g.node:
    for i in n.input:
        if i in init and not i.startswith('/'): used.setdefault(i,[]).append(n.name)
shared = {k:v for k,v in used.items() if len(v)>1}
print(f"  initializers referenced by >1 node: {len(shared)} of {len([k for k in init if not k.startswith('/')])}")
print(f"  e.g. downsampling.0.weight used by: {used.get('downsampling.0.weight')}")

print("\n-- activation tensor peak sizes (fp32 MB) --")
big=[]
for v in list(g.value_info)+list(g.output):
    d=[x.dim_value for x in v.type.tensor_type.shape.dim]
    if d and all(d): big.append((int(np.prod(d))*4/1e6, tuple(d), v.name))
big.sort(reverse=True)
for b,d,n in big[:8]: print(f"  {b:8.2f} MB {str(d):26s} {n}")
