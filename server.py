"""
Fluid-mixing simulation server.

POST /run               — runs the simulation with given params, returns run id
GET  /stream/<run_id>   — SSE stream of per-frame results
"""

import base64
import io
import json
import uuid

import numpy as np
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from PIL import Image

from analysis import (
    build_graph,
    compute_global_stats,
    compute_similarity_matrix,
    distinct_degree_types,
    frame_to_lab_features,
    resize_frame,
)
from simulation import Simulation

GIF_SIZE     = (128, 128)   # animated GIF (smaller keeps file size sane)
GIF_DURATION = 60           # ms per frame  (~16 fps)

app = Flask(__name__)
_runs: dict[str, dict] = {}  # run_id -> {results, gif}


def make_gif(rgb_frames: list) -> str:
    frames = [Image.fromarray(f).resize(GIF_SIZE, Image.NEAREST) for f in rgb_frames]
    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,              # loop forever
        duration=GIF_DURATION,
        optimize=True,
    )
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    p = request.json or {}
    n_steps    = max(10,  min(300, int(p.get("n_steps",    80))))
    diffusion  = max(0.01, min(0.24, float(p.get("diffusion",  0.08))))
    flow_speed = max(0.5,  min(6.0,  float(p.get("flow_speed", 2.0))))

    sim = Simulation(diffusion=diffusion, flow_speed=flow_speed)

    # ── Pass 1: run all steps, collect frames + exact entropy ──────────────
    rgb_frames = []
    entropies  = []
    for _ in range(n_steps):
        sim.step()
        rgb_frames.append(sim.to_rgb())
        entropies.append(sim.boltzmann_entropy())

    # ── Pass 2: compute global LAB stats, then complexity per frame ─────────
    small_frames = [resize_frame(f) for f in rgb_frames]
    all_features = [frame_to_lab_features(f) for f in small_frames]
    gmean, gstd  = compute_global_stats(all_features)

    results = []
    for i, (features, rgb, entropy) in enumerate(
        zip(all_features, rgb_frames, entropies)
    ):
        sim_matrix = compute_similarity_matrix(features, gmean, gstd)
        G          = build_graph(sim_matrix)
        results.append({
            "frame":      i + 1,
            "total":      n_steps,
            "entropy":    round(entropy, 5),
            "complexity": distinct_degree_types(G),
        })

    run_id = str(uuid.uuid4())
    _runs[run_id] = {"results": results, "gif": make_gif(rgb_frames)}
    return jsonify({"id": run_id, "total": n_steps})


@app.route("/stream/<run_id>")
def stream(run_id):
    stored = _runs.pop(run_id, None)
    if stored is None:
        return "Run ID not found", 404

    results = stored["results"]
    gif     = stored["gif"]

    def generate():
        yield f"data: {json.dumps({'type': 'total', 'total': results[-1]['total']})}\n\n"
        for r in results:
            yield f"data: {json.dumps({'type': 'frame', **r})}\n\n"
        yield f"data: {json.dumps({'type': 'gif',  'data': gif})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'total': results[-1]['total']})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


if __name__ == "__main__":
    app.run(debug=True, port=5002)
