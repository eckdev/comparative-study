import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.datasets.orthodontic_dataset import OrthodonticDataset


HTML_TEMPLATE = """<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    html, body {{ margin: 0; height: 100%; background: #111417; color: #e8ecef; font-family: Arial, sans-serif; }}
    #wrap {{ display: grid; grid-template-columns: minmax(0, 1fr) 320px; height: 100%; }}
    canvas {{ width: 100%; height: 100%; display: block; background: #111417; }}
    aside {{ border-left: 1px solid #2a3036; padding: 16px; overflow: auto; background: #171b20; }}
    h1 {{ font-size: 18px; margin: 0 0 8px; }}
    h2 {{ font-size: 13px; margin: 18px 0 8px; color: #aeb8c2; text-transform: uppercase; letter-spacing: .04em; }}
    .metric {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; font-size: 13px; padding: 6px 0; border-bottom: 1px solid #262c32; }}
    .legend {{ display: grid; gap: 7px; font-size: 13px; }}
    .sw {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 7px; vertical-align: -1px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    td, th {{ text-align: right; padding: 5px 3px; border-bottom: 1px solid #262c32; }}
    td:first-child, th:first-child {{ text-align: left; }}
    .hint {{ color: #aeb8c2; font-size: 12px; line-height: 1.45; }}
    a {{ color: #8fd0ff; }}
  </style>
</head>
<body>
<div id="wrap">
  <canvas id="c"></canvas>
  <aside>
    <h1>{title}</h1>
    <div class="hint">Sürükle: döndür · Mouse wheel: yakınlaştır · Çift tık: reset</div>
    <h2>Özet</h2>
    <div class="metric"><span>Sample</span><b>{sample_id}</b></div>
    <div class="metric"><span>Class / cinsiyet</span><b>{class_name} / {gender}</b></div>
    <div class="metric"><span>Sample ALE</span><b>{sample_ale:.3f}</b></div>
    <div class="metric"><span>Median hata</span><b>{sample_median:.3f}</b></div>
    <div class="metric"><span>Max hata</span><b>{sample_max:.3f}</b></div>
    <h2>Renkler</h2>
    <div class="legend">
      <div><span class="sw" style="background:#8b949e"></span>Yüz point cloud</div>
      <div><span class="sw" style="background:#33d17a"></span>Uzman landmark</div>
      <div><span class="sw" style="background:#4aa3ff"></span>PAL-Net tahmini</div>
      <div><span class="sw" style="background:#ffb86c"></span>Hata vektörü</div>
    </div>
    <h2>En yüksek landmark hataları</h2>
    {table}
  </aside>
</div>
<script>
const DATA = {data_json};
const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl', {{ antialias: true }});
if (!gl) alert('WebGL desteklenmiyor.');

function shader(type, src) {{
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}}
const vs = shader(gl.VERTEX_SHADER, `
attribute vec3 a_position;
attribute vec3 a_color;
attribute float a_size;
uniform mat4 u_matrix;
uniform float u_pointScale;
varying vec3 v_color;
void main() {{
  gl_Position = u_matrix * vec4(a_position, 1.0);
  gl_PointSize = a_size * u_pointScale;
  v_color = a_color;
}}`);
const fs = shader(gl.FRAGMENT_SHADER, `
precision mediump float;
varying vec3 v_color;
void main() {{
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  if (dot(p, p) > 1.0) discard;
  gl_FragColor = vec4(v_color, 1.0);
}}`);
const program = gl.createProgram();
gl.attachShader(program, vs);
gl.attachShader(program, fs);
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
gl.useProgram(program);

const locPos = gl.getAttribLocation(program, 'a_position');
const locColor = gl.getAttribLocation(program, 'a_color');
const locSize = gl.getAttribLocation(program, 'a_size');
const locMatrix = gl.getUniformLocation(program, 'u_matrix');
const locPointScale = gl.getUniformLocation(program, 'u_pointScale');

function makeBuffer(arr, itemSize, loc) {{
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr.flat()), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, itemSize, gl.FLOAT, false, 0, 0);
  return buf;
}}

const positions = [];
const colors = [];
const sizes = [];
function addPoints(points, color, size) {{
  for (const p of points) {{ positions.push(p); colors.push(color); sizes.push([size]); }}
}}
addPoints(DATA.points, [0.55,0.58,0.62], 1.15);
addPoints(DATA.expert, [0.20,0.82,0.48], 7.0);
addPoints(DATA.pred, [0.29,0.64,1.00], 7.0);

const linePositions = [];
const lineColors = [];
const lineSizes = [];
for (let i = 0; i < DATA.expert.length; i++) {{
  linePositions.push(DATA.expert[i], DATA.pred[i]);
  lineColors.push([1.0,0.72,0.42], [1.0,0.72,0.42]);
  lineSizes.push([1.0], [1.0]);
}}

let rotX = -0.15, rotY = 0.25, zoom = 2.35;
let dragging = false, lastX = 0, lastY = 0;
canvas.addEventListener('mousedown', e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }});
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('mousemove', e => {{
  if (!dragging) return;
  rotY += (e.clientX - lastX) * 0.008;
  rotX += (e.clientY - lastY) * 0.008;
  lastX = e.clientX; lastY = e.clientY;
  draw();
}});
canvas.addEventListener('wheel', e => {{ e.preventDefault(); zoom *= Math.exp(e.deltaY * 0.001); draw(); }}, {{ passive: false }});
canvas.addEventListener('dblclick', () => {{ rotX = -0.15; rotY = 0.25; zoom = 2.35; draw(); }});

function matMul(a,b) {{
  const r = new Array(16).fill(0);
  for (let c=0;c<4;c++) for (let r0=0;r0<4;r0++) for (let k=0;k<4;k++) r[c*4+r0] += a[k*4+r0]*b[c*4+k];
  return r;
}}
function perspective(fovy, aspect, near, far) {{
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return [f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,(2*far*near)*nf,0];
}}
function translate(z) {{ return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,z,1]; }}
function rotateX(a) {{ const c=Math.cos(a),s=Math.sin(a); return [1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]; }}
function rotateY(a) {{ const c=Math.cos(a),s=Math.sin(a); return [c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]; }}

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(canvas.clientWidth * dpr), h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {{ canvas.width = w; canvas.height = h; gl.viewport(0,0,w,h); }}
}}

function bindAndDraw(pos, col, siz, mode) {{
  makeBuffer(pos, 3, locPos);
  makeBuffer(col, 3, locColor);
  makeBuffer(siz, 1, locSize);
  gl.drawArrays(mode, 0, pos.length);
}}
function draw() {{
  resize();
  gl.clearColor(0.067,0.078,0.090,1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  const aspect = canvas.width / canvas.height;
  let m = perspective(Math.PI/4, aspect, 0.01, 100);
  m = matMul(m, translate(-zoom));
  m = matMul(m, rotateX(rotX));
  m = matMul(m, rotateY(rotY));
  gl.uniformMatrix4fv(locMatrix, false, new Float32Array(m));
  gl.uniform1f(locPointScale, Math.max(1.0, window.devicePixelRatio || 1));
  bindAndDraw(linePositions, lineColors, lineSizes, gl.LINES);
  bindAndDraw(positions, colors, sizes, gl.POINTS);
}}
window.addEventListener('resize', draw);
draw();
</script>
</body>
</html>
"""


def read_predictions(path):
    grouped = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["sample_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["landmark"]))
    return grouped


def normalize_scene(*arrays):
    stacked = np.concatenate(arrays, axis=0)
    center = stacked.mean(axis=0)
    scale = np.linalg.norm(stacked - center, axis=1).max()
    scale = scale if scale > 0 else 1.0
    return [((array - center) / scale).astype(float) for array in arrays]


def select_samples(grouped):
    values = []
    for sample_id, rows in grouped.items():
        errors = np.array([float(r["localization_error"]) for r in rows])
        values.append((float(errors.mean()), sample_id))
    values.sort()
    return [values[0][1], values[len(values) // 2][1], values[-1][1]]


def top_error_table(errors, limit=8):
    order = np.argsort(errors)[::-1][:limit]
    rows = ["<table><tr><th>LM</th><th>Hata</th></tr>"]
    for idx in order:
        rows.append(f"<tr><td>{int(idx)}</td><td>{errors[idx]:.3f}</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)


def write_colored_ply(path, points, expert, pred):
    vertices = []
    for p in points:
        vertices.append((*p, 140, 148, 158))
    for p in expert:
        vertices.append((*p, 51, 209, 122))
    for p in pred:
        vertices.append((*p, 74, 163, 255))

    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines.extend(f"{x} {y} {z} {r} {g} {b}" for x, y, z, r, g, b in vertices)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create 3D HTML comparison views for PAL-Net vs expert landmarks.")
    parser.add_argument("--data-root", default="../../data/dataset")
    parser.add_argument("--run-dir", default="../runs/orthodontic_palnet_procrustes_rigid_patch100_e40")
    parser.add_argument("--transformation-dir", default="../transforms/orthodontic_procrustes_rigid")
    parser.add_argument("--output-dir", default="../runs/orthodontic_palnet_procrustes_rigid_patch100_e40/visualizations")
    parser.add_argument("--sample-id", action="append", help="Sample id to render. May be repeated.")
    parser.add_argument("--max-points", type=int, default=25000)
    parser.add_argument("--surface-points", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = read_predictions(run_dir / "predictions_test.csv")
    sample_ids = args.sample_id or select_samples(grouped)

    dataset = OrthodonticDataset(
        args.data_root,
        cache_dir=None,
        num_surface_points=args.surface_points,
        transformation_dir=args.transformation_dir,
    )
    id_to_index = {sample.sample_id: idx for idx, sample in enumerate(dataset.samples)}

    index_rows = []
    for sample_id in sample_ids:
        if sample_id not in grouped:
            raise ValueError(f"{sample_id} is not present in predictions_test.csv")
        if sample_id not in id_to_index:
            raise ValueError(f"{sample_id} is not present in dataset")

        rows = grouped[sample_id]
        expert = np.array([[float(r["expert_x"]), float(r["expert_y"]), float(r["expert_z"])] for r in rows], dtype=np.float32)
        pred = np.array([[float(r["palnet_x"]), float(r["palnet_y"]), float(r["palnet_z"])] for r in rows], dtype=np.float32)
        errors = np.array([float(r["localization_error"]) for r in rows], dtype=np.float32)

        _, _, vertices = dataset[id_to_index[sample_id]]
        points = vertices.numpy()[:, :3].astype(np.float32)
        if len(points) > args.max_points:
            points = points[rng.choice(len(points), args.max_points, replace=False)]

        points_n, expert_n, pred_n = normalize_scene(points, expert, pred)
        meta = dataset.samples[id_to_index[sample_id]]
        title = f"{sample_id} | PAL-Net vs uzman"
        data = {
            "points": points_n.round(6).tolist(),
            "expert": expert_n.round(6).tolist(),
            "pred": pred_n.round(6).tolist(),
        }

        html = HTML_TEMPLATE.format(
            title=title,
            sample_id=sample_id,
            class_name=meta.class_name,
            gender=meta.gender,
            sample_ale=float(errors.mean()),
            sample_median=float(np.median(errors)),
            sample_max=float(errors.max()),
            table=top_error_table(errors),
            data_json=json.dumps(data),
        )
        html_path = output_dir / f"{sample_id}.html"
        html_path.write_text(html, encoding="utf-8")
        write_colored_ply(output_dir / f"{sample_id}_colored_points.ply", points, expert, pred)
        index_rows.append((sample_id, float(errors.mean()), html_path.name))

    index_lines = [
        "<!doctype html><html><head><meta charset='utf-8'><title>PAL-Net 3D Karşılaştırmalar</title>",
        "<style>body{font-family:Arial,sans-serif;background:#111417;color:#e8ecef;padding:24px}a{color:#8fd0ff}td,th{padding:8px 14px;border-bottom:1px solid #30363d;text-align:left}</style>",
        "</head><body><h1>PAL-Net 3D Karşılaştırmalar</h1><table><tr><th>Sample</th><th>ALE</th><th>Görsel</th></tr>",
    ]
    for sample_id, ale, filename in index_rows:
        index_lines.append(f"<tr><td>{sample_id}</td><td>{ale:.3f}</td><td><a href='{filename}'>{filename}</a></td></tr>")
    index_lines.append("</table></body></html>")
    (output_dir / "index.html").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"Wrote {len(index_rows)} visualizations to {output_dir}")
    print(f"Open: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
