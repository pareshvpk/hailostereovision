import numpy as np, onnxruntime as ort
np.random.seed(1)
p = r"C:\Microchip\hailo-stereo\artifacts\stereonet_pretrained\stereonet.onnx"
s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
H,W = 368,1232
# textured synthetic left image
base = np.random.rand(1,3,H,W+256).astype(np.float32)
L = base[:,:,:,128:128+W].copy()
def shift(dpx):  # right image = left shifted left by dpx  => true disparity = dpx
    return base[:,:,:,128+dpx:128+dpx+W].copy()

print("="*72); print("TEST 2 - does predicted disparity respond to TRUE disparity?"); print("="*72)
print(f"{'true disp (px)':>15s} {'pred mean':>12s} {'pred std':>10s} {'max|d vs disp=0|':>18s}")
ref=None
for dpx in [0,4,8,16,32,64,128]:
    out = s.run(None, {"input.1":L, "input.83":shift(dpx)})[0]
    if ref is None: ref = out
    print(f"{dpx:15d} {out.mean():12.5f} {out.std():10.5f} {np.abs(out-ref).max():18.6e}")

print("\n"+"="*72); print("TEST 3 - does the RIGHT image influence the output at all?"); print("="*72)
variants = {
  "right = left (disp 0)": L,
  "right = random noise ": np.random.rand(1,3,H,W).astype(np.float32),
  "right = all zeros    ": np.zeros((1,3,H,W),np.float32),
  "right = all ones     ": np.ones((1,3,H,W),np.float32),
}
outs={}
for k,v in variants.items():
    o = s.run(None, {"input.1":L, "input.83":v})[0]; outs[k]=o
    print(f"  {k}  ->  mean={o.mean():9.5f}  std={o.std():8.5f}  min={o.min():8.4f} max={o.max():8.4f}")
ks=list(outs)
print("\n  pairwise max|difference| in predicted disparity (pixels):")
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        print(f"    {ks[i]} vs {ks[j]} : {np.abs(outs[ks[i]]-outs[ks[j]]).max():.5f} px  (mean {np.abs(outs[ks[i]]-outs[ks[j]]).mean():.5f})")

print("\n"+"="*72); print("TEST 4 - does the LEFT image influence the output?"); print("="*72)
R0 = shift(8)
for k,v in {"left = textured":L, "left = zeros":np.zeros((1,3,H,W),np.float32), "left = noise":np.random.rand(1,3,H,W).astype(np.float32)}.items():
    o = s.run(None, {"input.1":v, "input.83":R0})[0]
    print(f"  {k:16s} -> mean={o.mean():9.5f} std={o.std():8.5f}")
