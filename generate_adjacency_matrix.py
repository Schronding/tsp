import argparse
import json
import math
from pathlib import Path


def load_graph(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_direct_matrix(graph: dict):
    nodes = graph["nodes"]
    node_ids = [n["id"] for n in nodes]
    index = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    matrix = [[math.inf] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 0.0

    for node in nodes:
        i = index[node["id"]]
        for edge in node["conexiones"]:
            j = index[edge["destino"]]
            w = float(edge["peso_w_horas"])
            if w < matrix[i][j]:
                matrix[i][j] = w

    return node_ids, matrix


def floyd_warshall(matrix):
    n = len(matrix)
    dist = [row[:] for row in matrix]
    for k in range(n):
        for i in range(n):
            dik = dist[i][k]
            if math.isinf(dik):
                continue
            for j in range(n):
                alt = dik + dist[k][j]
                if alt < dist[i][j]:
                    dist[i][j] = alt
    return dist


def to_json_safe(matrix):
    safe = []
    for row in matrix:
        safe_row = []
        for val in row:
            if math.isinf(val):
                safe_row.append(None)
            else:
                safe_row.append(round(val, 4))
        safe.append(safe_row)
    return safe


def extract_submatrix(node_ids, matrix, selected_ids):
    idx = {nid: i for i, nid in enumerate(node_ids)}
    selected_idx = [idx[nid] for nid in selected_ids]
    sub = [[matrix[i][j] for j in selected_idx] for i in selected_idx]
    return sub


def parse_subset(raw_subset: str, node_ids):
    if not raw_subset:
        return []
    selected = [x.strip().upper() for x in raw_subset.split(",") if x.strip()]
    unknown = [x for x in selected if x not in set(node_ids)]
    if unknown:
        raise ValueError(f"Unknown node ids in subset: {unknown}")
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Build direct and shortest-path adjacency matrices for TSP."
    )
    parser.add_argument(
        "--input",
        default="states_routes.json",
        help="Input graph JSON path (default: states_routes.json)",
    )
    parser.add_argument(
        "--output",
        default="states_adjacency_matrix.json",
        help="Output matrix JSON path (default: states_adjacency_matrix.json)",
    )
    parser.add_argument(
        "--subset",
        default="",
        help="Comma-separated node IDs for TSP subset, e.g. CDMX,PUE,VER",
    )
    args = parser.parse_args()

    graph = load_graph(Path(args.input))
    node_ids, direct = build_direct_matrix(graph)
    shortest = floyd_warshall(direct)

    result = {
        "metadata": {
            "description": "Adjacency matrices from states_routes graph",
            "node_count": len(node_ids),
            "weight_unit": "hours",
            "direct_matrix_note": "null means no direct edge",
            "shortest_matrix_note": "all-pairs shortest-path travel time",
        },
        "node_ids": node_ids,
        "direct_adjacency_matrix": to_json_safe(direct),
        "shortest_path_matrix": to_json_safe(shortest),
    }

    subset_ids = parse_subset(args.subset, node_ids)
    if subset_ids:
        subset_matrix = extract_submatrix(node_ids, shortest, subset_ids)
        result["subset_for_tsp"] = {
            "selected_node_ids": subset_ids,
            "shortest_path_submatrix": to_json_safe(subset_matrix),
        }

    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Generated {args.output} with {len(node_ids)} nodes")
    if subset_ids:
        print(f"Included subset_for_tsp for {len(subset_ids)} nodes: {subset_ids}")


if __name__ == "__main__":
    main()