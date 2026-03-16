from collections.abc import Collection

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import networkx as nx  # type: ignore


NODE_COLOR_TEMPLATE: dict[str, str] = {
    "genre": "lightblue",
    "writers": "y",
    "name": "salmon",
    "casts": "m",
    "directors": "c",
    "certificate": "olive",
    "person": "m",
    "combo": "k",
}

LEGEND_LABEL_MAP: dict[str, str] = {
    "name": "movie",
    "casts": "actor",
    "directors": "director",
    "writers": "writer",
}

EDGE_TYPE_INFO: dict[frozenset[str], tuple[str, str]] = {
    frozenset({"name", "genre"}): ("Had Genre", "steelblue"),
    frozenset({"casts", "name"}): ("Acted In", "violet"),
    frozenset({"directors", "name"}): ("Directed", "cyan"),
    frozenset({"writers", "name"}): ("Wrote", "gold"),
    frozenset({"name", "certificate"}): ("Is Rated", "tomato"),
}


def _get_node_color(node_type: str) -> str:
    if node_type == "name":
        return NODE_COLOR_TEMPLATE["name"]
    if "," in node_type:
        return NODE_COLOR_TEMPLATE["combo"]
    return NODE_COLOR_TEMPLATE.get(node_type, "gray")


def show_graph(
    G: nx.Graph,
    categories: Collection[str],
    title: str = "Graph from pandas dataframe",
) -> None:
    node_colors: list[str] = [_get_node_color(G.nodes[node]["node_type"]) for node in G.nodes()]

    plt.figure()
    pos = nx.nx_pydot.graphviz_layout(G, prog="neato")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.5, node_size=20)
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=0.5)

    legend_patches = [
        mpatches.Patch(color=color, label=LEGEND_LABEL_MAP.get(category, category))
        for category, color in NODE_COLOR_TEMPLATE.items()
        if category in categories
    ]
    if NODE_COLOR_TEMPLATE["combo"] in node_colors:
        legend_patches.append(
            mpatches.Patch(color=NODE_COLOR_TEMPLATE["combo"], label="combo")
        )

    plt.legend(handles=legend_patches, title="Node Types", loc="best")
    plt.title(title)
    plt.axis("off")


def show_graph_with_edge_colors(
    G: nx.Graph,
    categories: Collection[str],
    title: str = "Graph with colored edges",
) -> None:
    node_colors: list[str] = [_get_node_color(G.nodes[node]["node_type"]) for node in G.nodes()]

    schema_edge_list: list[tuple[int, int]] = []
    edge_colors: list[str] = []
    present_edge_types: dict[str, str] = {}
    for u, v in G.edges():
        type_u = G.nodes[u]["node_type"].split(",")[0]
        type_v = G.nodes[v]["node_type"].split(",")[0]
        pair = frozenset({type_u, type_v})
        if pair in EDGE_TYPE_INFO:
            label, color = EDGE_TYPE_INFO[pair]
            schema_edge_list.append((u, v))
            edge_colors.append(color)
            present_edge_types[label] = color

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(right=0.72)

    pos = nx.nx_pydot.graphviz_layout(G, prog="neato")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.5, node_size=20, ax=ax)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=schema_edge_list,
        edge_color=edge_colors,
        width=0.5,
        ax=ax,
    )

    node_legend_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor=NODE_COLOR_TEMPLATE[category],
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=8,
            label=LEGEND_LABEL_MAP.get(category, category),
        )
        for category in categories
        if category in NODE_COLOR_TEMPLATE
    ]
    node_legend = ax.legend(
        handles=node_legend_handles,
        title="Node Types",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
    )
    ax.add_artist(node_legend)

    edge_legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=color,
            linewidth=2,
            linestyle="-",
            solid_capstyle="butt",
            label=label,
            markersize=0,
        )
        for label, color in present_edge_types.items()
    ]
    ax.legend(
        handles=edge_legend_handles,
        title="Edge Types",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        borderaxespad=0.0,
    )

    ax.set_title(title)
    ax.axis("off")