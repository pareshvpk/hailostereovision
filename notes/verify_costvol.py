import onnx, numpy as np, onnxruntime as ort
from onnx import shape_inference
np.random.seed(0)
p = r"C:\Microchip\hailo-stereo\artifacts\stereonet_pretrained\stereonet.onnx"
m = onnx.load(p)
probe = [f"/Sub_{i}_output_0" for i in range(1,12)]
probe = ["/Sub_output_0"]+probe+["/Slice_output_0","/Slice_5_output_0","/res/res.6/Conv_output_0","/Squeeze_output_0"]
existing = {o.name for o in m.graph.output}
for name in probe:
    if name not in existing:
        m.graph.output.append(onnx.helper.ValueInfoProto(name=name))
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
L = np.random.rand(1,3,368,1232).astype(np.float32)
R = np.random.rand(1,3,368,1232).astype(np.float32)
outs = sess.run(None, {"input.1":L, "input.83":R})
names = [o.name for o in sess.get_outputs()]
d = dict(zip(names,outs))

print("="*72); print("TEST 1 — are the 12 cost-volume disparity slices distinct?"); print("="*72)
base = d["/Sub_output_0"]
for i in range(1,12):
    s = d[f"/Sub_{i}_output_0"]
    print(f"  Sub_{i:<2d} (disparity level {i:2d})  max|diff vs level 0| = {np.abs(s-base).max():.6e}   identical={np.array_equal(s,base)}")
print("\n  Slice_0 == left features (unshifted)? ", np.array_equal(d["/Slice_output_0"], d["/res/res.6/Conv_output_0"]))
print("  Slice_5 == left features (unshifted)? ", np.array_equal(d["/Slice_5_output_0"], d["/res/res.6/Conv_output_0"]))
cv = d["/Squeeze_output_0"]
print(f"\n  filtered cost volume {cv.shape}: per-disparity std across channels = {cv.std(axis=1).mean():.6e}")
print(f"  cost volume channel means: {np.round(cv.mean(axis=(0,2,3)),5)}")
