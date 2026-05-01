import time
import pickle
import argparse


import numpy as np
from tqdm import tqdm
import networkx as nx
import graph_tool.all as gt
from syntherela.metadata import Metadata

from nx2gt import nx2gt, gt2nx


def create_graph_without_edges(graph):
    """
    Creates a new graph_tool graph with the same vertices as the input graph but without any edges.

    Args:
      graph: The input graph_tool graph object.

    Returns:
      A new graph_tool graph with the same vertices but no edges.
    """

    # Create a new graph with the same directedness as the input graph
    new_graph = gt.Graph(directed=graph.is_directed())

    # Add all vertices from the original graph to the new graph
    for _ in range(graph.num_vertices()):
        new_graph.add_vertex()

    for key in graph.vertex_properties:
        new_graph.vertex_properties[key] = new_graph.new_vertex_property(
            graph.vp[key].value_type()
        )
        for v in new_graph.vertices():
            new_graph.vp[key][v] = graph.vp[key][graph.vertex(int(v))]

    return new_graph


def generate_new_graph(
    g, micro_degs=True, micro_ers=True, hstate=None, max_retries=10000
):
    v = create_graph_without_edges(g)

    if hstate is None:
        print("Minimizing the H-SBM")
        hstate = gt.minimize_nested_blockmodel_dl(
            g, state_args={"deg_corr": True, "clabel": g.vp["block"]}
        )
    else:
        print("Using the provided H-SBM")
        print(hstate)
    blockstate = hstate.levels[0]

    fk_prop = v.new_edge_property("string")
    v.edge_properties["type"] = fk_prop

    edge_types = set(g.ep["type"])

    pbar = tqdm(edge_types, total=len(edge_types))
    for fk in pbar:
        pbar.set_description(f"Generating graph for relationship {fk}")
        # Select the bipartite subgraph induced by the relationship
        edge_filter = g.new_edge_property("bool")
        for edge in g.edges():
            edge_filter[edge] = g.ep["type"][edge] == fk
        h = gt.GraphView(g, efilt=edge_filter)

        # initialize the blockstate for the subgraph based on b*
        state = gt.BlockState(h, b=blockstate.b.a, deg_corr=True, clabel=g.vp["block"])

        has_parallel_edges = (
            gt.label_parallel_edges(h, mark_only=True).a.sum().item() > 0
        )

        h_ = gt.generate_sbm(
            b=state.b.a,
            probs=gt.adjacency(state.get_bg(), state.get_ers()).T,
            out_degs=h.degree_property_map("out").a,
            in_degs=h.degree_property_map("in").a,
            directed=h.is_directed(),
            micro_ers=micro_ers,
            micro_degs=micro_degs,
        )
        if not has_parallel_edges:
            parallel_edges = gt.label_parallel_edges(h_, mark_only=True).a
            num_parallel_edges = parallel_edges.sum().item()
            if num_parallel_edges > 0:
                min_parallel_edges = num_parallel_edges
                bar = tqdm(
                    range(max_retries),
                    total=max_retries,
                    desc=f"Retrying {num_parallel_edges} parallel edges in {fk} subgraph",
                )
                for i in bar:
                    bar.update(1)
                    h_ = gt.generate_sbm(
                        b=state.b.a,
                        probs=gt.adjacency(state.get_bg(), state.get_ers()).T,
                        out_degs=h.degree_property_map("out").a,
                        in_degs=h.degree_property_map("in").a,
                        directed=h.is_directed(),
                        micro_ers=micro_ers,
                        micro_degs=micro_degs,
                    )
                    parallel_edges = gt.label_parallel_edges(h_, mark_only=True).a
                    num_parallel_edges = parallel_edges.sum().item()
                    bar.set_description(f"Retrying - {fk} -({min_parallel_edges})")
                    if num_parallel_edges == 0:
                        h = h_
                        break
                    else:
                        min_parallel_edges = min(num_parallel_edges, min_parallel_edges)
                else:
                    # Rewire the graph to ensure it is simple.
                    print(
                        f"Max retries reached for {num_parallel_edges} parallel edges in {fk} subgraph switching to rewiring."
                    )

                    def edge_probs(r, s):
                        return state.get_matrix()[s, r]

                    gt.random_rewire(
                        h,
                        model="blockmodel-degree",
                        block_membership=state.b,
                        edge_probs=edge_probs,
                        verbose=True,
                        configuration=True,
                        self_loops=False,
                    )
            else:
                h = h_
        else:
            h = h_

        # Add the edges from the subgraph
        for edge in h.edges():
            e = v.add_edge(edge.source(), edge.target())
            v.ep["type"][e] = fk
    return v


def preprocess(g, fk_only_tables, split_by_subgraphs=False, stub_tables=None):
    keep = g.new_vertex_property("bool")
    reverse_fk = dict()
    if fk_only_tables is None:
        fk_only_tables = []
    if stub_tables is None:
        stub_tables = []
    else:
        stub_counts = dict()
        for table in stub_tables:
            stub_counts[table] = g.new_vertex_property("int")
            g.vp[f"{table}_count"] = stub_counts[table]

    for node in tqdm(g.vertices(), total=g.num_vertices(), desc="Preprocessing"):
        node_type = g.vp["type"][node]
        if node_type in fk_only_tables:
            if g.is_directed():
                n1, n2 = list(node.in_neighbours())
            else:
                n1, n2 = list(node.all_neighbours())

            table = g.vp["type"][node]
            keep[node] = False

            parent1 = g.vp["type"][n1]
            parent2 = g.vp["type"][n2]
            reverse_fk.setdefault((table, parent1), list())
            reverse_fk.setdefault((table, parent2), list())
            fk1 = g.ep["type"][g.edge(n1, node)]
            fk2 = g.ep["type"][g.edge(n2, node)]
            keys = {n1: fk1, n2: fk2}
            # sort the keys by the fk
            keys = dict(sorted(keys.items(), key=lambda item: item[1]))
            n1, n2 = keys.keys()
            fk1, fk2 = keys.values()

            edge = g.add_edge(n1, n2)
            g.ep["type"][edge] = table

            if fk1 not in reverse_fk[(table, parent1)]:
                reverse_fk[(table, parent1)].append(fk1)
            if fk2 not in reverse_fk[(table, parent2)]:
                reverse_fk[(table, parent2)].append(fk2)
        elif node_type in stub_tables:
            # get nodes neighbours
            if g.is_directed():
                n1 = list(node.in_neighbours())
            else:
                n1 = list(node.all_neighbours())
            assert len(n1) == 1, (
                f"Stub table {node_type} has more than one neighbour {n1}"
            )
            parent = n1[0]
            # increase the stub count for the parent node
            stub_counts[node_type][parent] = stub_counts[node_type][parent] + 1
            # remove the stub node from the graph
            keep[node] = False
        else:
            keep[node] = True

    h = gt.GraphView(g, vfilt=keep)
    h = gt.Graph(h, prune=True)

    # create new node property with an int for each table
    table_names = set(h.vp["type"])
    table_map = {table_name: i for i, table_name in enumerate(table_names)}

    table_prop = h.new_vertex_property("int")
    h.vertex_properties["t"] = table_prop
    for v in h.vertices():
        table_prop[v] = table_map[h.vp["type"][v]]

    # Split the nodes into blocks
    # Sgplit by tables
    blockpartition = table_prop.copy()
    if split_by_subgraphs:
        h.set_directed(False)
        comps, _ = gt.label_components(h)
        h.set_directed(True)
        print(np.unique(comps.a).shape[0], "disjoint subgraphs")
        # increment the partition by the number of tables times the subgraph index
        blockpartition.a += len(table_names) * comps.a

    h.vp["block"] = blockpartition
    return h, blockpartition, reverse_fk


def postprocess(g, fk_only_tables, metadata, reverse_fk, stub_tables=None):
    if stub_tables is not None:
        vertices = list(g.vertices())
        stub_fks = dict()
        for stub_table in stub_tables:
            parents = metadata.get_parents(stub_table)
            assert len(parents) == 1, (
                f"Stub table {stub_table} has more than one parent {parents}"
            )
            parent = parents.pop()
            foreign_keys = metadata.get_foreign_keys(parent, stub_table)
            assert len(foreign_keys) == 1, (
                f"Stub table {stub_table} has more than one foreign key {foreign_keys}"
            )
            stub_fks[stub_table] = foreign_keys.pop()

        for node in tqdm(vertices, desc="Postprocessing"):
            for stub_table in stub_tables:
                if g.vp[f"{stub_table}_count"][node] > 0:
                    # add the stub nodes to the graph
                    for _ in range(g.vp[f"{stub_table}_count"][node]):
                        new_node = g.add_vertex()
                        g.vp["type"][new_node] = stub_table
                        # add the edge to the parent node
                        edge = g.add_edge(node, new_node)
                        # add the foreign key to the edge
                        fk = stub_fks[stub_table]
                        g.ep["type"][edge] = (parent, fk, stub_table)

    if fk_only_tables is None:
        return g
    foreign_keys = dict()
    for table in fk_only_tables:
        for parent in metadata.get_parents(table):
            foreign_keys[(parent, table)] = reverse_fk[(table, parent)]

    # transform back to the original graph
    edges = list(g.edges())
    for edge in tqdm(edges, total=g.num_edges(), desc="Postprocessing"):
        table = g.ep["type"][edge]

        if table in fk_only_tables:
            node = g.add_vertex()
            g.vp["type"][node] = table
            # Add edge to the first parent
            src = edge.source()
            edge1 = g.add_edge(src, node)
            parent1 = g.vp["type"][src]
            # Add edge to the second parent
            tgt = edge.target()
            edge2 = g.add_edge(tgt, node)
            parent2 = g.vp["type"][tgt]
            if parent1 == parent2:
                fk1, fk2 = foreign_keys[(parent1, table)]
            else:
                fk1 = foreign_keys[(parent1, table)][0]
                fk2 = foreign_keys[(parent2, table)][0]
            g.ep["type"][edge1] = fk1
            g.ep["type"][edge2] = fk2
            # Remove the edge
            g.remove_edge(edge)

    return g


def sort_nodes_by_table(G, copy=True, table_order=None):
    # sort nodes by table
    table_to_nodes = dict()
    for node in G.nodes:
        table = G.nodes[node]["type"]
        if table not in table_to_nodes:
            table_to_nodes[table] = []
        table_to_nodes[table].append(node)

    mapping = dict()
    i = 0
    if table_order is None:
        table_order = sorted(table_to_nodes.keys())

    for table in table_order:
        print(table, i)
        for node in table_to_nodes[table]:
            mapping[node] = i
            i += 1

    return nx.relabel_nodes(G, mapping, copy=copy)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_name", default="f1_subsampled", type=str)
    argparser.add_argument("--data-dir", default="data", type=str)
    args = argparser.parse_args()

    dataset = args.dataset_name
    data_dir = args.data_dir

    with open(f"{data_dir}/structure/{dataset}_graph.pkl", "rb") as f:
        G = pickle.load(f)

    metadata = Metadata().load_from_json(f"{data_dir}/original/{dataset}/metadata.json")

    has_multiedges = False
    # Avoid modeling certain tables with 1:M relationships as these are maintained anyway.
    # This can significantly speed up the generation process by avoiding redundant rewiring.
    stub_tables = None
    if dataset == "ccs_clava":
        stub_tables = ["yearmonth"]
        fk_tables = None
        split_by_subgraph = False
    elif dataset == "imdb_MovieLens_v1":
        fk_tables = ["movies2actors", "movies2directors", "u2base"]
        split_by_subgraph = False
    elif dataset == "movie_lens_clava":
        fk_tables = ["movie2actor", "movie2director", "rating"]
        split_by_subgraph = False
    elif dataset == "Biodegradability_v1":
        fk_tables = ["bond", "gmember"]
        split_by_subgraph = True
    elif dataset == "CORA_v1" or dataset == "CORA_updated":
        has_multiedges = True
        fk_tables = ["cites"]
        if dataset == "CORA_updated":
            fk_tables = ["cites", "content"]
        split_by_subgraph = False
    elif dataset == "berka_clava" or dataset == "Berka_subsampled":
        fk_tables = None
        split_by_subgraph = False
    elif dataset == "instacart_05_clava":
        fk_tables = None
        split_by_subgraph = False
    elif dataset == "f1_subsampled":
        fk_tables = None
        split_by_subgraph = False
    else:
        fk_tables = None
        split_by_subgraph = True

    # Convert the NetworkX graph to a graph-tool format
    g = nx2gt(G)

    # Transform the many-to-many tables into edges and trim stub tables
    h, blocks, reverse_fk = preprocess(
        g,
        fk_tables,
        split_by_subgraphs=split_by_subgraph,
        stub_tables=stub_tables,
    )
    start_time = time.time()
    # Find the maximum-likelihood partition of the graph
    state = gt.minimize_nested_blockmodel_dl(
        h, state_args={"deg_corr": True, "clabel": h.vp["block"]}
    )
    end_time = time.time()
    print(f"Time taken to learn the H-SBM: {end_time - start_time:.2f} seconds")

    start_time = time.time()
    # Sample the new graph which preserves the block structure and joint degree distribution
    u = generate_new_graph(
        h, micro_degs=True, micro_ers=True, hstate=state, max_retries=10
    )
    end_time = time.time()
    print(f"Time taken to generate the new graph: {end_time - start_time:.2f} seconds")

    # Revert the preprocessing transformations
    u = postprocess(u, fk_tables, metadata, reverse_fk, stub_tables=stub_tables)

    U = gt2nx(u, multiedges=has_multiedges)

    # Order the nodes for consistency with the pyg dataset.
    U = sort_nodes_by_table(U)

    # Ensure the joint degree distribution is preserved
    nkk = nx.assortativity.degree_mixing_dict(U, x="out", y="in", normalized=False)
    nkk_orig = nx.assortativity.degree_mixing_dict(G, x="out", y="in", normalized=False)
    assert nkk_orig == nkk

    # Save the generated graph
    with open(f"{data_dir}/structure/{dataset}_graph_gen.pkl", "wb") as f:
        pickle.dump(U, f)
