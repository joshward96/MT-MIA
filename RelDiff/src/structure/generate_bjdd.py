import pickle
import argparse

import networkx as nx
from tqdm import tqdm
from syntherela.metadata import Metadata

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_name", default="f1_subsampled", type=str)
    argparser.add_argument("--data-dir", default="data", type=str)
    args = argparser.parse_args()

    dataset = args.dataset_name
    data_dir = args.data_dir

    # Load the original graph (generate it with to_networkx.py) and metadata
    with open(f"{data_dir}/structure/{dataset}_graph.pkl", "rb") as f:
        G: nx.DiGraph = pickle.load(f)

    metadata = Metadata().load_from_json(f"{data_dir}/original/{dataset}/metadata.json")

    edge_types = sorted(list(set(nx.get_edge_attributes(G, "type").values())))

    # create empty graph with same nodes without edges
    G_syn = nx.DiGraph()
    G_syn.add_nodes_from(G.nodes(data=True))

    for edge_type in edge_types:
        print(edge_type)
        # select the edges that correspond to the current edge type (foreign key / m2m relationship)
        selected_edges = [
            (u, v) for u, v, e in G.edges(data=True) if e["type"] == edge_type
        ]
        # select the induced bipartite graph for the current edge type
        B = G.edge_subgraph(selected_edges)
        print(B.number_of_nodes(), B.number_of_edges())

        # Compute the directed joint degree matrix
        in_degrees = [deg for node, deg in B.in_degree()]
        out_degrees = [deg for node, deg in B.out_degree()]
        nkk = nx.assortativity.degree_mixing_dict(B, x="out", y="in", normalized=False)

        assert nx.is_valid_directed_joint_degree(
            in_degrees=in_degrees, out_degrees=out_degrees, nkk=nkk
        )

        # Sample a directed graph with the same joint degree distribution
        B_syn = nx.directed_joint_degree_graph(
            in_degrees=in_degrees, out_degrees=out_degrees, nkk=nkk
        )

        node_mapping = {i: node for i, node in enumerate(B.nodes())}

        for edge in tqdm(B_syn.edges(data=True), desc="Adding edges"):
            u, v = edge[0], edge[1]
            assert u in node_mapping, f"Node {u} not found in mapping"
            assert v in node_mapping, f"Node {v} not found in mapping"
            u_orig = node_mapping[u]
            v_orig = node_mapping[v]
            assert B_syn.degree(u) == B.degree(u_orig) and B_syn.degree(v) == B.degree(
                v_orig
            ), f"Degree mismatch for nodes {u} and {v}"
            assert (
                G_syn.nodes[u_orig]["type"] == edge_type[0]
                and G_syn.nodes[v_orig]["type"] == edge_type[2]
            ), f"Type mismatch for nodes {u_orig} and {v_orig}"
            G_syn.add_edge(u_orig, v_orig, type=edge_type)

    print("Original graph")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    print("Synthetic graph")
    print("Nodes:", G_syn.number_of_nodes())
    print("Edges:", G_syn.number_of_edges())

    # Ensure the joint degree distribution matches.
    nkk_orig = nx.assortativity.degree_mixing_dict(G, x="out", y="in", normalized=False)
    nkk_syn = nx.assortativity.degree_mixing_dict(
        G_syn, x="out", y="in", normalized=False
    )

    assert nkk_orig == nkk_syn, "Joint degree distribution mismatch"

    # Save the synthetic graph
    with open(f"{data_dir}/structure/{dataset}_graph_2k.pkl", "wb") as f:
        pickle.dump(G_syn, f)
