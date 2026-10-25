import os
from dataset_loader import load_graph_cache
from mof_graph import load_structure, build_bonded_graph, find_bridging_carboxylate_path
from gnccp_matcher import gnccp_match

_THIS_DIR = os.path.dirname(os.path.abspath('performance_report.py'))
graphs = load_graph_cache(os.path.join(_THIS_DIR, 'qmof_graph_cache.json'))

G_mof5 = load_structure(os.path.join(_THIS_DIR, 'data', 'MOF-5.cif'))
G_mof5 = build_bonded_graph(G_mof5)
Gq_zn, _ = find_bridging_carboxylate_path(G_mof5, 'Zn')

print(f'Query Motif Extracted: Zn-carboxylate (from MOF-5), Size: {Gq_zn.number_of_nodes()} atoms')
print('\n--- 2D GNCCP SCORES ACROSS ALL 7 MOFs ---')

results = []
for name, Hc in graphs.items():
    if Hc.number_of_nodes() < Gq_zn.number_of_nodes():
        continue
    try:
        res = gnccp_match(Gq_zn, Hc)
        results.append((name, res['score'], res['mapping']))
        print(f'{name:15s} | 2D Similarity Score: {res["score"]:.4f}')
    except Exception as e:
        print(f'{name:15s} | Error: {e}')

results.sort(key=lambda x: x[1], reverse=True)
best_name, best_score, best_map = results[0]

print(f'\nBest Match: {best_name} with score {best_score:.4f}')
print('Mapping:')
for q_n, c_n in list(best_map.items()):
    q_elem = Gq_zn.nodes[q_n]['element']
    c_elem = graphs[best_name].nodes[c_n]['element']
    print(f'  Motif Node {q_n} ({q_elem}) -> {best_name} Node {c_n} ({c_elem})')
