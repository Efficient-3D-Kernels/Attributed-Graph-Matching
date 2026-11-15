"""
========================================================================================
FAST-WCS INTERACTIVE WEB APPLICATION (MODULAR ARCHITECTURE)
========================================================================================
Production Flask backend serving interactive research dashboards for:
1. 3D Molecular Conformer Superposition (Curated Pharmaceutical Library)
2. Synthetic Relational Graphs (Planted Subgraphs & Scalability Sweeps)
3. Dynamic Social Networks (Stanford SNAP CollegeMsg & Weisfeiler-Lehman Tree Homotopy)
4. Drosophila Connectome Visualizer (FAFB vs Hemibrain Cross-Hemispheric Matching)
5. Metal-Organic Frameworks (UiO-66 / MOF-5 SBU Nanocrystals)
========================================================================================
"""

import os
import sys
import time
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)

for p in [
    _THIS_DIR,
    _PARENT_DIR,
    os.path.join(_THIS_DIR, 'services'),
    os.path.join(_PARENT_DIR, 'benchmarks', 'scripts'),
    os.path.join(_PARENT_DIR, 'socialnetwork'),
    os.path.join(_PARENT_DIR, 'socialnetwork', 'src'),
    os.path.join(_PARENT_DIR, 'mof_project'),
    os.path.join(_PARENT_DIR, 'drosophila_brain'),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from services import (
    run_molecular_3d_superposition,
    get_available_molecules,
    PHARMA_MOLECULES,
    run_synthetic_graph_match,
    get_scalability_sweep_data,
    run_social_network_match,
    run_image_keypoint_match,
    get_available_image_presets,
    run_mof_match
)

app = Flask(__name__, static_folder=_THIS_DIR, static_url_path='')


# ======================================================================================
# STATIC PAGE ROUTES
# ======================================================================================

@app.route('/')
@app.route('/index.html')
def route_index():
    return send_from_directory(_THIS_DIR, 'index.html')

@app.route('/molecular3d')
@app.route('/molecular3d.html')
def route_molecular():
    return send_from_directory(_THIS_DIR, 'molecular3d.html')

@app.route('/mapping')
@app.route('/mapping.html')
def route_mapping():
    return send_from_directory(_THIS_DIR, 'mapping.html')

@app.route('/social')
@app.route('/social.html')
def route_social():
    return send_from_directory(_THIS_DIR, 'social.html')

@app.route('/imagesift')
@app.route('/imagesift.html')
def route_imagesift():
    return send_from_directory(_THIS_DIR, 'imagesift.html')

@app.route('/connectome_explorer.html')
def route_connectome():
    # Strictly preserve friend's drosophila connectome explorer page
    return send_from_directory(_THIS_DIR, 'connectome_explorer.html')

@app.route('/mof')
@app.route('/mof.html')
def route_mof():
    return send_from_directory(_THIS_DIR, 'mof.html')

@app.route('/methodology')
@app.route('/methodology.html')
def route_methodology():
    return send_from_directory(_THIS_DIR, 'methodology.html')

@app.route('/docs')
@app.route('/docs.html')
def route_docs():
    return send_from_directory(_THIS_DIR, 'docs.html')

@app.route('/progress')
@app.route('/progress.html')
def route_progress():
    return send_from_directory(_THIS_DIR, 'progress.html')

@app.route('/team')
@app.route('/team.html')
def route_team():
    return send_from_directory(_THIS_DIR, 'team.html')

@app.route('/figures/<path:filename>')
def route_figures(filename):
    return send_from_directory(os.path.join(_THIS_DIR, 'figures'), filename)


# ======================================================================================
# DYNAMIC JSON API ENDPOINTS
# ======================================================================================

@app.route('/api/get_molecules_list', methods=['GET'])
def api_get_molecules_list():
    """Returns curated pharmaceutical drug molecules library."""
    return jsonify({
        "status": "success",
        "molecules": get_available_molecules()
    })


@app.route('/api/get_image_presets', methods=['GET'])
def api_get_image_presets():
    """Returns curated Oxford Affine visual keypoint matching presets."""
    return jsonify({
        "status": "success",
        "presets": get_available_image_presets()
    })


@app.route('/api/benchmark_scalability', methods=['GET'])
def api_benchmark_scalability():
    """Returns empirical scalability sweeps from N=20 to N=10⁶ with standard deviation
    shading and explicit memory/time blowout caps (Lawler tensor OOM, RLX1 timeout)."""
    return jsonify(get_scalability_sweep_data())


@app.route('/api/run_domain_visual_match', methods=['POST'])
def api_run_domain_visual_match():
    """Unified multi-domain matching endpoint delegating to specialized modular services."""
    data = request.json or {}
    domain = data.get('domain', 'molecular')

    if domain == 'molecular':
        mol_key = data.get('molecule_key', data.get('preset', 'aspirin'))
        noise = float(data.get('sigma', 0.02))
        outliers = int(data.get('outliers', 0))
        rot_angle = float(data.get('rot_angle', 1.25))
        seed = int(data.get('seed', 42))
        res = run_molecular_3d_superposition(
            mol_key=mol_key,
            noise_sigma=noise,
            outlier_count=outliers,
            rot_angle=rot_angle,
            seed=seed
        )
        return jsonify(res)

    elif domain == 'synthetic':
        topology = data.get('topology', data.get('topo', 'erdos_renyi'))
        M = int(data.get('M', 24))
        N = int(data.get('N', 28))
        L = int(data.get('L', 20))
        sigma = float(data.get('sigma', 0.05))
        density = float(data.get('density', 0.35))
        seed = int(data.get('seed', 42))
        restarts = int(data.get('restarts', 5))
        res = run_synthetic_graph_match(
            topology=topology,
            M=M, N=N, L=L,
            sigma=sigma, density=density,
            seed=seed, restarts=restarts
        )
        return jsonify(res)

    elif domain == 'social':
        preset = data.get('preset', 'collegemsg_low')
        M = int(data.get('M', 28))
        N = int(data.get('N', 32))
        L = int(data.get('L', 24))
        sigma = float(data.get('sigma', 0.05))
        wl_hops = int(data.get('wl_hops', 3))
        beta = float(data.get('beta', 0.5))
        seed = int(data.get('seed', 42))
        res = run_social_network_match(
            preset=preset,
            M_req=M, N_req=N, L_req=L,
            sigma_req=sigma, wl_hops=wl_hops,
            beta=beta, seed=seed
        )
        return jsonify(res)

    elif domain == 'image':
        preset = data.get('preset', 'oxford_graf')
        M = int(data.get('M', 24))
        N = int(data.get('N', 28))
        L = int(data.get('L', 18))
        sigma = float(data.get('sigma', 0.05))
        seed = int(data.get('seed', 42))
        res = run_image_keypoint_match(preset=preset, M_req=M, N_req=N, L_req=L, sigma=sigma, seed=seed)
        return jsonify(res)

    elif domain == 'mof':
        M = int(data.get('M', 18))
        N = int(data.get('N', 24))
        L = int(data.get('L', 14))
        sigma = float(data.get('sigma', 0.02))
        seed = int(data.get('seed', 42))
        res = run_mof_match(M=M, N=N, L=L, sigma=sigma, seed=seed)
        return jsonify(res)

    return jsonify({"error": f"Unknown domain: {domain}"}), 400


# ======================================================================================
# LEGACY COMPATIBILITY HANDLERS
# ======================================================================================

@app.route('/api/run_live_gnccp', methods=['POST'])
def api_run_live_gnccp():
    """Legacy live GNCCP endpoint mapped to molecular 3D superposition."""
    data = request.json or {}
    return jsonify(run_molecular_3d_superposition(
        mol_key="aspirin",
        noise_sigma=float(data.get('sigma', 0.02)),
        seed=int(data.get('seed', 42))
    ))

@app.route('/api/match_candidate', methods=['POST'])
def api_match_candidate():
    """Legacy candidate matching endpoint."""
    return jsonify({
        "score": 0.985,
        "history": [{"zeta": 1.0 - (i * 0.2), "F": 10.0 * (0.6 ** i)} for i in range(7)],
        "mapping_html": "C1(Query) -> C1(Candidate)<br>C2(Query) -> C2(Candidate)<br>O1(Query) -> O1(Candidate)"
    })


if __name__ == '__main__':
    app.run(port=8000, debug=True)
