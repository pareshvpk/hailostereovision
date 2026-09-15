"""
Interactive inspection server for HailoStereo.

    python src/demo_server.py --ckpt runs/kitti_border/best.pt
    then open http://localhost:8000

Upload a rectified stereo pair and watch it move through the network: the two
inputs, the matching features, what the cost volume actually believes at each
disparity, the confidence of the match, the refinement ladder rung by rung, and
the final disparity -- with the time each stage took.

The point is that the intermediate stages are visible. The model this replaces
produced a plausible-looking disparity map from a cost volume that compared
every hypothesis at zero shift; nothing downstream of that was doing stereo, and
you could not tell from the output alone. Here the cost slices and the
confidence map show whether matching is happening, not just what came out.

Standard library only -- no Flask, no Gradio. Uploads arrive as base64 JSON
rather than multipart, which keeps the server to one small handler.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo, MAX_DISP, NUM_DISP, MATCH_SCALE  # noqa: E402
from data import MEAN, STD  # noqa: E402
from preview import DISP_RAMP, ERR_RAMP, ramp, to_u8  # noqa: E402

HEIGHT, WIDTH = 368, 1232
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# image handling
# ---------------------------------------------------------------------------

def prepare(img: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """Any image -> the graph's 368x1232 input, plus the RGB it was made from.

    Scaled to width and bottom-cropped, matching src/make_calib.py so what the
    demo shows is what the calibration set and the device would see.
    """
    img = img.convert("RGB")
    scale = WIDTH / img.width
    img = img.resize((WIDTH, max(HEIGHT, round(img.height * scale))), Image.BILINEAR)
    img = img.crop((0, img.height - HEIGHT, WIDTH, img.height))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1))[None]
    return x, rgb


def png_b64(arr_u8: np.ndarray, scale: float = 0.5) -> str:
    im = Image.fromarray(arr_u8)
    if scale != 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                       Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# visualisations
# ---------------------------------------------------------------------------

def features_rgb(f: torch.Tensor) -> np.ndarray:
    """Project a feature map to three channels by PCA, so it can be looked at.

    The three principal directions of the 32-channel matching feature carry most
    of what the cost volume compares. Colour here means "similar features", not
    depth -- two pixels the same colour are ones the matcher would consider
    interchangeable.
    """
    x = f[0].detach().float().cpu().numpy()          # [C,H,W]
    c, h, w = x.shape
    flat = x.reshape(c, -1)
    flat = flat - flat.mean(1, keepdims=True)
    # economy SVD on the channel covariance: 32x32, trivial
    u, _, _ = np.linalg.svd(flat @ flat.T)
    proj = (u[:, :3].T @ flat).reshape(3, h, w)
    lo = proj.reshape(3, -1).min(1)[:, None, None]
    hi = proj.reshape(3, -1).max(1)[:, None, None]
    return to_u8(((proj - lo) / np.maximum(hi - lo, 1e-6)).transpose(1, 2, 0))


def disp_rgb(d: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return to_u8(ramp((d - lo) / max(hi - lo, 1e-6), DISP_RAMP))


def prob_rgb(p: np.ndarray) -> np.ndarray:
    """One disparity hypothesis' probability, on the error ramp (dark = likely)."""
    return to_u8(ramp(p, ERR_RAMP))


# ---------------------------------------------------------------------------
# staged inference
# ---------------------------------------------------------------------------

class Timer:
    """Wall time per stage, with the CUDA queue drained first.

    Without the synchronise every stage but the last would report near zero:
    kernel launches return immediately and the work happens later. That is the
    classic way to publish a latency number that is off by an order of magnitude.
    """

    def __init__(self):
        self.marks = []
        self._t = self._now()

    @staticmethod
    def _now():
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def lap(self, name):
        t = self._now()
        self.marks.append((name, 1000.0 * (t - self._t)))
        self._t = t


@torch.no_grad()
def run_stages(model, left, right, left_rgb, right_rgb):
    """Run the network stage by stage, collecting panels and timings."""
    left, right = left.to(DEVICE), right.to(DEVICE)

    # One untimed pass first. A GPU sitting idle is at its low clock state and
    # takes a few hundred milliseconds to boost, so whichever stage runs first
    # absorbs the ramp: an early version of this timed the two identical feature
    # towers at 22.6 ms and 7.2 ms. The numbers below are warm, which is also
    # what a device serving a stream would see.
    model(left, right)
    timer = Timer()

    fr2, fr4, fr8 = model.features(right)
    timer.lap("features (right)")
    fl2, fl4, fl8 = model.features(left)
    timer.lap("features (left)")

    raw = model.cost_volume(fl8, fr8)
    timer.lap("cost volume")
    cost = model.aggregation(raw)
    timer.lap("aggregation")

    prob = F.softmax(-cost * model.soft_argmin.temperature(), dim=1)
    d8 = model.soft_argmin.index(prob)
    timer.lap("soft-argmin")

    d4 = model.refine4(d8, fl4)
    timer.lap("refine 1/4")
    d2 = model.refine2(d4, fl2)
    timer.lap("refine 1/2")
    d1 = F.relu(model.refine1(d2, left))
    timer.lap("refine 1/1")

    final = d1[0, 0].cpu().numpy()
    lo, hi = float(np.percentile(final, 2)), float(np.percentile(final, 98))

    panels = [
        ("Left input", png_b64(to_u8(left_rgb)),
         "The reference view. Disparity is predicted in this frame."),
        ("Right input", png_b64(to_u8(right_rgb)),
         "The same scene from the second camera. Every match is a horizontal "
         "shift from here into the left view."),
        ("Matching features, left (1/8)", png_b64(features_rgb(fl8)),
         "The 32-channel feature the matcher compares, projected to RGB by PCA. "
         "Colour means similar-to-the-matcher, not depth."),
        ("Matching features, right (1/8)", png_b64(features_rgb(fr8)),
         "The same projection on the right view. Corresponding points should "
         "carry the same colour, offset horizontally by their disparity."),
    ]

    # what the cost volume actually believes, at four disparities
    p = prob[0].cpu().numpy()
    for d in (0, NUM_DISP // 3, 2 * NUM_DISP // 3, NUM_DISP - 1):
        panels.append((
            f"P(disparity = {d * MATCH_SCALE} px)",
            png_b64(prob_rgb(p[d]), scale=1.0),
            f"Probability mass on hypothesis {d} of {NUM_DISP}. Dark is likely. "
            f"These four should look DIFFERENT -- near surfaces light up in the "
            f"high-disparity slices, far ones in the low. Identical slices mean "
            f"the cost volume is not searching."))

    ent = -(p * np.log(p + 1e-9)).sum(0)
    panels.append((
        "Match confidence", png_b64(to_u8(1.0 - ramp(ent / np.log(NUM_DISP), ERR_RAMP)), scale=1.0),
        f"Softmax entropy, inverted: bright is a confident match. Textureless "
        f"road and sky are uncertain, edges are sharp. Uniform entropy of "
        f"{np.log(NUM_DISP):.2f} would mean no hypothesis is preferred."))

    for name, t, note in (
        ("Disparity 1/8 (from matching)", d8,
         "Straight out of the soft-argmin. This is the only stage that does "
         "stereo; everything after it refines."),
        ("Refined 1/4", d4, "First rung of the ladder, guided by left-image features."),
        ("Refined 1/2", d2, "Second rung."),
    ):
        a = t[0, 0].cpu().numpy() * (MAX_DISP / NUM_DISP if t is d8 else
                                     (4.0 if t is d4 else 2.0))
        panels.append((name, png_b64(disp_rgb(a, lo, hi)), note))

    panels.append((
        "Final disparity (full res)", png_b64(disp_rgb(final, lo, hi), scale=1.0),
        f"Light is near, dark is far. Range {final.min():.1f}-{final.max():.1f} px, "
        f"shown on the 2nd-98th percentile ({lo:.1f}-{hi:.1f} px)."))

    # The array itself, so the page can report numbers and probe a pixel.
    # Half resolution, uint16 at 1/64 px: 226 KB instead of 1.8 MB of float32,
    # and 1/64 px is far finer than the model resolves.
    half = np.ascontiguousarray(final[::2, ::2])
    q = np.clip(half * 64.0, 0, 65535).astype("<u2")
    disp = {"w": int(q.shape[1]), "h": int(q.shape[0]), "scale": 64.0,
            "data": base64.b64encode(q.tobytes()).decode(),
            "min": float(final.min()), "max": float(final.max()),
            "median": float(np.median(final)),
            "p2": lo, "p98": hi}
    return panels, timer.marks, final, disp


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>HailoStereo inspector</title><style>
:root{--bg:#fbfbfa;--fg:#1a1a19;--mut:#6b6b68;--line:#e3e2df;--card:#fff;--accent:#2a5599}
@media(prefers-color-scheme:dark){:root{--bg:#191918;--fg:#eeeeec;--mut:#9a9a96;--line:#33322f;--card:#222220;--accent:#8fb3e8}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:20px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--mut);margin:0 0 24px}
.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;padding:16px;background:var(--card);
border:1px solid var(--line);border-radius:10px;margin-bottom:20px}
label.f{font-size:13px;color:var(--mut);display:flex;flex-direction:column;gap:4px}
input[type=file]{font:12px system-ui;max-width:230px}
button{font:14px system-ui;padding:8px 16px;border-radius:7px;border:1px solid var(--line);
background:var(--card);color:var(--fg);cursor:pointer}
button.pri{background:var(--accent);color:#fff;border-color:transparent}
button:disabled{opacity:.5;cursor:default}
#msg{color:var(--mut);margin:14px 0}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;margin:0 0 26px;min-width:340px}
th,td{text-align:left;padding:5px 16px 5px 0;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
td.n{text-align:right}
.bar2{display:inline-block;height:8px;background:var(--accent);border-radius:2px;vertical-align:middle}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.card img{width:100%;display:block;background:#000}
.card .t{font-weight:600;padding:10px 12px 2px}
.card .d{color:var(--mut);font-size:12.5px;padding:0 12px 12px}
.total{font-size:15px;margin:0 0 18px}
.total b{font-size:22px}
.res{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:0 0 20px}
.res h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);margin:0 0 12px;font-weight:600}
.kv{display:flex;flex-wrap:wrap;gap:26px}
.kv div{min-width:96px}
.kv .k{font-size:12px;color:var(--mut)}
.kv .v{font-size:19px;font-variant-numeric:tabular-nums}
.probe{margin-top:14px;padding-top:12px;border-top:1px solid var(--line);color:var(--mut);font-size:13px}
.probe b{color:var(--fg);font-size:16px;font-variant-numeric:tabular-nums}
#pimg{cursor:crosshair;max-width:100%;display:block;border-radius:6px;margin-top:10px}
</style></head><body><div class="wrap">
<h1>HailoStereo inspector</h1>
<p class="sub">Upload a rectified stereo pair, or load a KITTI sample. Every
stage of the network is shown, with the time it took.</p>
<div class="bar">
  <label class="f">Left image<input type="file" id="L" accept="image/*"></label>
  <label class="f">Right image<input type="file" id="R" accept="image/*"></label>
  <button id="run" class="pri">Run</button>
  <button id="samp">Load KITTI sample</button>
  <span style="flex:1"></span>
  <label class="f">Focal length (px)<input type="number" id="foc" value="715.7" step="0.1" style="width:100px"></label>
  <label class="f">Baseline (m)<input type="number" id="base" value="0.54" step="0.01" style="width:80px"></label>
</div>
<div id="msg"></div>
<div id="depth"></div>
<div id="out"></div>
</div><script>
let L=null,R=null;
const $=i=>document.getElementById(i), msg=t=>$('msg').textContent=t;
function read(f){return new Promise(r=>{const x=new FileReader();x.onload=()=>r(x.result);x.readAsDataURL(f)})}
$('L').onchange=async e=>{if(e.target.files[0])L=await read(e.target.files[0])};
$('R').onchange=async e=>{if(e.target.files[0])R=await read(e.target.files[0])};
$('samp').onclick=async()=>{
  msg('loading a sample pair...');
  try{const r=await fetch('/sample?i='+Math.floor(Math.random()*40));
    if(!r.ok)throw new Error(await r.text());
    const j=await r.json();L=j.left;R=j.right;msg('sample loaded - press Run');
  }catch(e){msg('could not load a sample: '+e.message)}
};
$('run').onclick=async()=>{
  if(!L||!R){msg('need both a left and a right image');return}
  $('run').disabled=true;msg('running...');$('out').innerHTML='';$('depth').innerHTML='';D=null;
  try{
    const r=await fetch('/infer',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({left:L,right:R})});
    if(!r.ok)throw new Error(await r.text());
    render(await r.json());msg('');
  }catch(e){msg('failed: '+e.message)}
  $('run').disabled=false;
};
let D=null;
function depthOf(px){const f=+$('foc').value,b=+$('base').value;
  return (px>0.05&&f>0&&b>0)?f*b/px:null}
function fmtZ(z){return z===null?'--':(z<1000?z.toFixed(1)+' m':'far')}
function renderDepth(){
  if(!D){$('depth').innerHTML='';return}
  const near=depthOf(D.p98), far=depthOf(D.p2), mid=depthOf(D.median);
  $('depth').innerHTML=
   '<div class="res"><h2>Result</h2><div class="kv">'+
   '<div><div class="k">disparity range</div><div class="v">'+D.min.toFixed(1)+' - '+D.max.toFixed(1)+' px</div></div>'+
   '<div><div class="k">median disparity</div><div class="v">'+D.median.toFixed(1)+' px</div></div>'+
   '<div><div class="k">nearest (98th pct)</div><div class="v">'+fmtZ(near)+'</div></div>'+
   '<div><div class="k">median depth</div><div class="v">'+fmtZ(mid)+'</div></div>'+
   '<div><div class="k">farthest (2nd pct)</div><div class="v">'+fmtZ(far)+'</div></div>'+
   '</div><div class="probe">Depth uses <b>Z = f x B / d</b> with the focal length and baseline '+
   'in the toolbar - currently '+(+$('foc').value)+' px and '+(+$('base').value)+' m, KITTI 2015 defaults. '+
   'They are assumptions, not measurements: for your own camera pair they come from calibration, and '+
   'a wrong value scales every depth below by the same factor. Disparity is what the model actually '+
   'predicts.<br><br><span style="color:#c25a4a">Known defect: above the horizon the depth is wrong.</span> '+
   'KITTI's LiDAR ground truth does not reach the sky, so that region was never supervised during '+
   'finetuning and the model drifted there - it reports the sky at ~71 px disparity (near) where the '+
   'SceneFlow pretrain gave ~14 px (far). Neither eval protocol scores it, because neither has ground '+
   'truth there. Trust the lower two thirds of the frame.'+
   '<br><br><b id="pv">Click the map to read a point</b>'+
   '<img id="pimg" src="'+D.img+'"></div></div>';
  $('pimg').onclick=e=>{
    const r=e.target.getBoundingClientRect();
    const x=Math.floor((e.clientX-r.left)/r.width*D.w), y=Math.floor((e.clientY-r.top)/r.height*D.h);
    if(x<0||y<0||x>=D.w||y>=D.h)return;
    const d=D.a[y*D.w+x]/D.scale, z=depthOf(d);
    $('pv').textContent='x '+(x*2)+', y '+(y*2)+'  ->  disparity '+d.toFixed(2)+' px  ->  depth '+fmtZ(z);
  };
}
$('foc').oninput=renderDepth; $('base').oninput=renderDepth;
function render(j){
  const raw=atob(j.disp.data), buf=new Uint8Array(raw.length);
  for(let i=0;i<raw.length;i++)buf[i]=raw.charCodeAt(i);
  D=Object.assign({},j.disp,{a:new Uint16Array(buf.buffer)});
  D.img=(j.panels.find(p=>p[0].startsWith('Final disparity'))||[])[1];
  renderDepth();
  const mx=Math.max(...j.timings.map(t=>t[1]));
  let h='<p class="total">Total <b>'+j.total_ms.toFixed(1)+' ms</b> on '+j.device+
        ' &middot; '+(1000/j.total_ms).toFixed(1)+' fps &middot; '+j.shape+
        '<br><span style="font-size:12.5px;color:var(--mut)">Warm: an untimed pass runs first, '+
        'so the GPU is at boost clocks and the first stage is not charged for the ramp.</span></p>';
  h+='<table><tr><th>stage</th><th>ms</th><th>share</th></tr>';
  for(const [n,ms] of j.timings)
    h+='<tr><td>'+n+'</td><td class="n">'+ms.toFixed(2)+'</td><td><span class="bar2" style="width:'+
       Math.max(2,160*ms/mx)+'px"></span></td></tr>';
  h+='</table><div class="grid">';
  for(const p of j.panels)
    h+='<div class="card"><img src="'+p[1]+'"><div class="t">'+p[0]+'</div><div class="d">'+p[2]+'</div></div>';
  $('out').innerHTML=h+'</div>';
}
</script></body></html>"""


def make_handler(model, samples):
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):        # quiet: one line per request is noise
            pass

        def _send(self, code, body, ctype="application/json"):
            body = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/sample"):
                if not samples:
                    self._send(404, "no dataset found on disk to sample from",
                               "text/plain")
                    return
                try:
                    i = int(self.path.split("i=")[1]) % len(samples)
                except (IndexError, ValueError):
                    i = 0
                lp, rp = samples[i][0], samples[i][1]
                out = {}
                for key, path in (("left", lp), ("right", rp)):
                    with open(path, "rb") as fh:
                        raw = fh.read()
                    kind = "png" if str(path).lower().endswith(".png") else "webp"
                    out[key] = f"data:image/{kind};base64," + \
                        base64.b64encode(raw).decode()
                self._send(200, json.dumps(out))
                return
            self._send(200, PAGE, "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/infer":
                self._send(404, "not found", "text/plain")
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n))
                imgs = []
                for key in ("left", "right"):
                    b64 = req[key].split(",", 1)[-1]
                    imgs.append(Image.open(io.BytesIO(base64.b64decode(b64))))
                (lx, lrgb), (rx, rrgb) = (prepare(i) for i in imgs)
                panels, marks, _, disp = run_stages(model, lx, rx, lrgb, rrgb)
                self._send(200, json.dumps({
                    "panels": panels, "timings": marks, "disp": disp,
                    "total_ms": sum(m for _, m in marks),
                    "device": DEVICE, "shape": f"{HEIGHT}x{WIDTH}"}))
            except Exception as exc:      # surfaced in the page, not just the console
                self._send(500, f"{type(exc).__name__}: {exc}", "text/plain")

    return H


def find_samples():
    """A few real pairs to try, if either dataset is on disk."""
    root = pathlib.Path(__file__).resolve().parent.parent
    k = root / "data" / "kitti2015" / "training"
    if (k / "image_2").is_dir():
        lefts = sorted((k / "image_2").glob("*_10.png"))[-40:]
        return [(l, k / "image_3" / l.name) for l in lefts
                if (k / "image_3" / l.name).exists()]
    d = root / "data" / "driving" / "frames"
    if d.is_dir():
        lefts = sorted(d.glob("*/*/*/left/*.webp"))[::97][:40]
        return [(l, pathlib.Path(str(l).replace("left", "right"))) for l in lefts]
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/kitti_border/best.pt")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = HailoStereo()
    model.load_state_dict(blob.get("model", blob), strict=True)   # EXP-1
    model.to(DEVICE).eval()
    print(f"{args.ckpt}: epoch {blob.get('epoch','?')}, "
          f"val EPE {blob.get('epe', float('nan')):.3f} px on {DEVICE}")

    samples = find_samples()
    print(f"{len(samples)} sample pairs available" if samples else
          "no dataset on disk -- upload your own pair")

    # one warm-up: the first CUDA call pays for context creation and cuDNN
    # autotuning, which would otherwise be charged to the user's first upload
    with torch.no_grad():
        z = torch.zeros(1, 3, HEIGHT, WIDTH, device=DEVICE)
        model(z, z)
    print(f"ready: http://localhost:{args.port}")

    from http.server import ThreadingHTTPServer
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model, samples)) \
        .serve_forever()


if __name__ == "__main__":
    main()
