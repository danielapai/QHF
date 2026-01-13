"""
QHF Interactive Dashboard Application

This module provides a web-based dashboard for visualizing Quantitative Habitability Framework (QHF)
results. It includes interactive graphs, distribution plots, scatter plots, and video content.

Features:
- Interactive module dependency graph with node clicking and information display
- Distribution histograms and 3D scatter plots
- Pairwise relationship scatter plots
- Font size controls for graph readability
- Video tutorial section
- Home page with QHF framework information

Author: QHF Development Team
"""

# ======================================
# IMPORTS
# ======================================

import json
from pathlib import Path
import numpy as np
import networkx as nx
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os, base64
from dash import Dash, dcc, html, Input, Output, State, callback_context, clientside_callback, ClientsideFunction


# ======================================
# DATA LOADING FUNCTIONS
# ======================================
def find_results_file():
    """
    Locate the most recent results JSON file from common directory locations.
    
    Returns:
        Path: Path object pointing to the most recent results file
        
    Raises:
        FileNotFoundError: If no results file is found in any candidate location
    """
    candidates = [
        Path("../results/latest.json"),
        Path("results/latest.json"),
        Path("../../results/latest.json"),
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError("No results/latest.json found. Run QHF once.")
    return max(existing, key=lambda p: p.stat().st_mtime)


def load_data():
    """
    Load QHF results data from the latest JSON file.
    
    Returns:
        dict: QHF results data including graph, distributions, and metadata
    """
    p = find_results_file()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["_results_path"] = str(p.resolve())
    return data


# ======================================
# DATA CLEANING AND UTILITY FUNCTIONS
# ======================================
def clean_1d(x):
    """
    Remove NaN and infinite values from a 1D array.
    
    Args:
        x: Input array or list
        
    Returns:
        list: Cleaned list with only finite values
    """
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)].tolist()


def safe_clean_xy(x, y):
    """
    Clean and align two arrays, removing pairs with NaN or infinite values.
    
    Args:
        x: First array
        y: Second array
        
    Returns:
        tuple: (cleaned_x, cleaned_y) as lists
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return [], []
    a = a[:n]; b = b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    return a[m].tolist(), b[m].tolist()


def safe_clean_xyz(x, y, z):
    """
    Clean and align three arrays, removing triplets with NaN or infinite values.
    
    Args:
        x: First array
        y: Second array
        z: Third array
        
    Returns:
        tuple: (cleaned_x, cleaned_y, cleaned_z) as lists
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    c = np.asarray(z, dtype=float).ravel()
    n = min(a.size, b.size, c.size)
    if n == 0:
        return [], [], []
    a = a[:n]; b = b[:n]; c = c[:n]
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    return a[m].tolist(), b[m].tolist(), c[m].tolist()


def ulabel(label, unit):
    """
    Format a label with its unit in brackets.
    
    Args:
        label: Parameter name
        unit: Unit string
        
    Returns:
        str: Formatted label with unit, or just label if unit is unitless
    """
    if unit and str(unit).lower() not in ("unitless", "-", "—", "none"):
        return f"{label} [{unit}]"
    return label


# ======================================
# ZOOM AND VIEW CONTROL FUNCTIONS
# ======================================
def _fig_bounds(fig):
    """
    Calculate the bounding box of all data points in a figure.
    
    Args:
        fig: Plotly figure dictionary
        
    Returns:
        tuple: (xmin, xmax, ymin, ymax) with 5% padding
    """
    xs, ys = [], []
    for tr in (fig or {}).get("data", []):
        xs.extend([float(v) for v in (tr.get("x") or []) if v is not None])
        ys.extend([float(v) for v in (tr.get("y") or []) if v is not None])
    if not xs or not ys:
        return -1, 1, -1, 1
    pad_x = 0.05 * (max(xs) - min(xs) or 1.0)
    pad_y = 0.05 * (max(ys) - min(ys) or 1.0)
    return min(xs) - pad_x, max(xs) + pad_x, min(ys) - pad_y, max(ys) + pad_y


def _apply_zoom(fig, factor=None, reset=False):
    """
    Apply zoom transformation to a Plotly figure.
    
    Args:
        fig: Plotly figure dictionary
        factor: Zoom factor (< 1.0 zooms in, > 1.0 zooms out)
        reset: If True, reset to auto-range
        
    Returns:
        dict: Updated figure dictionary
    """
    if fig is None:
        return fig
    fig = dict(fig)
    lay = fig.setdefault("layout", {})
    xa = lay.setdefault("xaxis", {})
    ya = lay.setdefault("yaxis", {})
    if reset:
        xa["autorange"] = True
        ya["autorange"] = True
        xa.pop("range", None)
        ya.pop("range", None)
        return fig
    if "range" in xa and "range" in ya:
        xmin, xmax = xa["range"]
        ymin, ymax = ya["range"]
    else:
        xmin, xmax, ymin, ymax = _fig_bounds(fig)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    hw = max(0.5 * (xmax - xmin), 1e-6)
    hh = max(0.5 * (ymax - ymin), 1e-6)
    if factor is None:
        return fig
    hw *= factor
    hh *= factor
    xa["autorange"] = False
    ya["autorange"] = False
    xa["range"] = [cx - hw, cx + hw]
    ya["range"] = [cy - hh, cy + hh]
    return fig


# ======================================
# FIGURE BUILDING FUNCTIONS
# ======================================
def build_graph_fig(data, node_positions=None, font_size=14):
    """
    Build an interactive Plotly graph figure showing module dependencies.
    
    Args:
        data: QHF results data dictionary
        node_positions: Optional dictionary of node positions to use
        font_size: Base font size for labels and text
        
    Returns:
        go.Figure: Plotly figure with nodes, edges, arrows, and planet image
    """
    gdata = data.get("graph", {}) or {}
    nodes = gdata.get("nodes", []) or []
    edges = gdata.get("edges", []) or []
    pos = node_positions if node_positions else (gdata.get("positions", {}) or {})
    directed = bool(gdata.get("directed", True))

    node_color_map = gdata.get("node_colors", {}) or {}
    default_color = "#1f77b4"
    
    # Override specific nodes to red
    red_nodes = ["stellar properties", "orbital parameters", "albedo prior"]
    # Override specific nodes to green
    green_nodes = ["methanogens AE v1.0"]
    
    # Check for case-insensitive matches, handling newlines and whitespace
    for node_name in nodes:
        # Normalize node name: lowercase, replace newlines with spaces, strip whitespace
        node_normalized = node_name.lower().replace("\n", " ").replace("\r", " ").strip()
        # Remove extra spaces
        node_normalized = " ".join(node_normalized.split())
        
        # Check for red nodes
        for red_node in red_nodes:
            red_normalized = red_node.lower().strip()
            # Check if normalized strings match or contain each other
            if red_normalized in node_normalized or node_normalized in red_normalized:
                node_color_map[node_name] = "red"
                break
        else:
            # Check for green nodes (only if not already set to red)
            for green_node in green_nodes:
                green_normalized = green_node.lower().strip()
                # Check if normalized strings match or contain each other
                if green_normalized in node_normalized or node_normalized in green_normalized:
                    node_color_map[node_name] = "green"
                    break
    
    node_colors = [node_color_map.get(n, default_color) for n in nodes]

    if not pos:
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from([(u, v) for (u, v, _lab) in edges])
        # Use kamada_kawai layout for better network graph appearance
        try:
            layout = nx.kamada_kawai_layout(G, seed=42)
        except:
            # Fallback to spring layout if kamada_kawai fails
            layout = nx.spring_layout(G, seed=42, k=1.5, iterations=50)
        pos = {n: [float(layout[n][0]), float(layout[n][1])] for n in G.nodes()}

    # Build node descriptions from edges
    node_info = {}
    for node in nodes:
        incoming = [e[0] for e in edges if e[1] == node]
        outgoing = [e[1] for e in edges if e[0] == node]
        edge_labels_in = [e[2] for e in edges if e[1] == node]
        edge_labels_out = [e[2] for e in edges if e[0] == node]
        node_info[node] = {
            "name": node.replace("\n", " ").strip(),
            "incoming": incoming,
            "outgoing": outgoing,
            "incoming_labels": edge_labels_in,
            "outgoing_labels": edge_labels_out
        }

    edge_x, edge_y = [], []
    for (u, v, _lab) in edges:
        if u not in pos or v not in pos:
            continue
        try:
            pos_u = pos[u]
            pos_v = pos[v]
            # Ensure positions are lists/tuples with 2 elements
            if not isinstance(pos_u, (list, tuple)) or len(pos_u) < 2:
                continue
            if not isinstance(pos_v, (list, tuple)) or len(pos_v) < 2:
                continue
            x0, y0 = float(pos_u[0]), float(pos_u[1])
            x1, y1 = float(pos_v[0]), float(pos_v[1])
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        except (ValueError, TypeError, IndexError):
            continue

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=2, color="rgba(100,149,237,0.6)", dash="solid"),
        hoverinfo="none",
        showlegend=False
        # Note: edges don't support selected/unselected styling in Plotly
    )

    node_x, node_y = [], []
    hover_texts = []
    valid_nodes = []
    valid_colors = []
    
    for i, n in enumerate(nodes):
        if n in pos:
            try:
                pos_n = pos[n]
                # Ensure position is a list/tuple with 2 elements
                if isinstance(pos_n, (list, tuple)) and len(pos_n) >= 2:
                    x, y = float(pos_n[0]), float(pos_n[1])
                    node_x.append(x)
                    node_y.append(y)
                    valid_nodes.append(n)
                    valid_colors.append(node_colors[i] if i < len(node_colors) else "#1f77b4")
                    # Create detailed hover text with node info
                    info = node_info.get(n, {})
                    hover = f"<b>{info.get('name', n)}</b><br>"
                    hover += f"<br>Click for detailed information<br>"
                    incoming_count = len(info.get('incoming', []))
                    outgoing_count = len(info.get('outgoing', []))
                    if incoming_count > 0:
                        hover += f"<br>Receives from: {incoming_count} module(s)"
                    if outgoing_count > 0:
                        hover += f"<br>Sends to: {outgoing_count} module(s)"
                    hover_texts.append(hover)
            except (ValueError, TypeError, IndexError):
                continue
    
    # Create node trace with enhanced interactivity
    if len(node_x) == 0:
        # No valid nodes, create empty trace
        node_trace = go.Scatter(x=[], y=[], mode="markers")
    else:
        # Prepare text for nodes (shortened versions for display)
        node_texts = [n.replace("\n", " ")[:20] + ("..." if len(n.replace("\n", " ")) > 20 else "") for n in valid_nodes]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            marker=dict(
                size=25,  # Larger nodes for better visibility
                color=valid_colors,
                line=dict(width=2, color="#1a1a1a"),
                opacity=0.9,
                sizemode="diameter"
            ),
            text=node_texts,  # Show shortened node names on nodes
            textposition="middle center",
            textfont=dict(size=max(10, int(font_size * 0.7)), color="black"),
            hovertext=hover_texts,
            hoverinfo="text",
            customdata=valid_nodes,  # Store node names for click detection
            selected=dict(
                marker=dict(size=35, opacity=1.0, color="#FFD700")  # Gold highlight on selection (size, opacity, color only)
            ),
            unselected=dict(marker=dict(opacity=0.7)),  # Dim unselected nodes
            showlegend=False
        )

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title=dict(
            text="QHF Module Dependency Graph (Click nodes for info)",
            font=dict(size=max(16, int(font_size * 1.1)), color="black", family="Arial, sans-serif"),
            x=0.5,
            xanchor="center"
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            visible=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        plot_bgcolor="white",  # White background for better visibility
        paper_bgcolor="white",  # White paper background
        dragmode="pan",  # Allow panning
        hovermode="closest",
        clickmode="event+select",  # Enable click and select modes
        selectdirection="any",  # Allow selection in any direction (valid values: 'h', 'v', 'd', 'any')
        # Enhanced interactivity settings
        uirevision="constant",  # Maintain UI state during updates
    )

    # arrows with labels
    if directed and edges:
        anns = []
        for (u, v, edge_label) in edges:
            if u not in pos or v not in pos:
                continue
            try:
                pos_u = pos[u]
                pos_v = pos[v]
                if not isinstance(pos_u, (list, tuple)) or len(pos_u) < 2:
                    continue
                if not isinstance(pos_v, (list, tuple)) or len(pos_v) < 2:
                    continue
                x0, y0 = float(pos_u[0]), float(pos_u[1])
                x1, y1 = float(pos_v[0]), float(pos_v[1])
                xm, ym = (x0 + x1)/2, (y0 + y1)/2
                dx, dy = (x1 - x0), (y1 - y0)
                
                # Position label on the line (no perpendicular offset)
                # Create arrow annotation with label positioned on the line
                ann = dict(
                    x=xm, y=ym,  # Position label on the line center
                    ax=xm - 0.1*dx, ay=ym - 0.1*dy,  # Arrow positioned closer to source
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
                    arrowcolor="rgba(100,149,237,0.8)",
                    text=str(edge_label) if edge_label else "",  # Add edge label text
                    xanchor="center", yanchor="middle",
                    font=dict(size=max(10, int(font_size * 0.75)), color="#1a1a1a", family="Arial"),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="rgba(100,149,237,0.8)", borderwidth=1.5, borderpad=4
                )
                anns.append(ann)
            except (ValueError, TypeError, IndexError):
                continue
        if anns:
            fig.update_layout(annotations=anns)

    # Node labels are now displayed directly on nodes via text in Scatter trace
    # No need for separate label annotations
    label_ann = []
    for n in nodes:
        if n not in pos:
            continue
        try:
            pos_n = pos[n]
            if isinstance(pos_n, (list, tuple)) and len(pos_n) >= 2:
                x, y = float(pos_n[0]), float(pos_n[1])
                # Node labels are now shown directly on nodes via text in Scatter
                # Keep this for additional labels if needed, but make them optional
                # label_ann.append(dict(
                #     x=x, y=y + 0.05, text=n, xref="x", yref="y", showarrow=False,
                #     xanchor="center", yanchor="bottom",
                #     font=dict(size=font_size, color="#111"),
                #     bgcolor="rgba(255,255,255,0.98)",
                #     bordercolor="#2c3e50", borderwidth=0.6, borderpad=2
                # ))
        except (ValueError, TypeError, IndexError):
            continue
    if label_ann:
        existing_anns = fig.layout.annotations or []
        fig.update_layout(annotations=list(existing_anns) + label_ann)

    # planet image
    meta = data.get("meta", {}) or {}
    pimg = meta.get("planet_image")
    planet_name = meta.get("habitat_shortname", "")
    
    if pimg and os.path.isfile(pimg):
        try:
            with open(pimg, "rb") as f:
                enc = base64.b64encode(f.read()).decode("ascii")
            fig.add_layout_image(dict(
                source=f"data:image/png;base64,{enc}",
                xref="paper", yref="paper", x=0.98, y=0.98,
                sizex=0.16, sizey=0.16, xanchor="right", yanchor="top", layer="above"
            ))
            # Add planet name below the image - add it after all other annotations
            if planet_name:
                # Get existing annotations and add planet name
                existing_anns = list(fig.layout.annotations or []) if fig.layout.annotations else []
                # Image is anchored at top-right (0.98, 0.98) with size 0.16
                # Image bottom is at y = 0.98 - 0.16 = 0.82
                # Place text below the image
                planet_ann = dict(
                    x=0.90, y=0.78,  # Position below the image (image bottom is at ~0.82)
                    xref="paper", yref="paper",
                    xanchor="center", yanchor="top",
                    text=str(planet_name),
                    showarrow=False,
                    font=dict(size=14, color="#111", family="Arial, sans-serif"),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#2c3e50", borderwidth=1, borderpad=4
                )
                existing_anns.append(planet_ann)
                fig.update_layout(annotations=existing_anns)
        except Exception:
            pass
    return fig


def build_distributions_fig(data):
    """
    Build a figure showing distribution histograms and 3D scatter plot.
    
    Args:
        data: QHF results data dictionary
        
    Returns:
        go.Figure: Plotly figure with histograms and 3D scatter
    """
    import math
    dist = data.get("distributions", {}) or {}
    T = dist.get("temperature", []) or []
    D = dist.get("depth", []) or []
    S = dist.get("suitability", []) or []

    labels = (data.get("meta", {}) or {}).get("labels", {}) or {}
    TL, TU = (labels.get("temperature", {}).get("label", "Temperature"),
              labels.get("temperature", {}).get("unit", "K"))
    DL, DU = (labels.get("depth", {}).get("label", "Depth"),
              labels.get("depth", {}).get("unit", "m"))
    SL, SU = (labels.get("suitability", {}).get("label", "Suitability"),
              labels.get("suitability", {}).get("unit", "unitless"))

    T, D, S = clean_1d(T), clean_1d(D), clean_1d(S)

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("Temperature Histogram",
                        f"{DL} Histogram",
                        "Suitability Histogram",
                        "3D: Temperature vs Depth vs Suitability")
    )

    if T:
        fig.add_trace(go.Histogram(x=T), row=1, col=1)
    if D:
        fig.add_trace(go.Histogram(x=D), row=1, col=2)
    if S:
        fig.add_trace(go.Histogram(x=S), row=2, col=1)

    Xt, Yt, Zt = safe_clean_xyz(T, D, S)
    if Xt and Yt and Zt:
        fig.add_trace(go.Scatter3d(x=Xt, y=Yt, z=Zt, mode="markers",
                                   marker=dict(size=3)), row=2, col=2)

    fig.update_xaxes(title_text=ulabel(TL, TU), row=1, col=1)
    fig.update_xaxes(title_text=ulabel(DL, DU), row=1, col=2)
    fig.update_xaxes(title_text=ulabel(SL, SU), row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    # 3D scatter axis names
    fig.update_scenes(
        dict(
            xaxis_title=ulabel(TL, TU),
            yaxis_title=ulabel(DL, DU),
            zaxis_title=ulabel(SL, SU)
        ),
        row=2, col=2
    )

    fig.update_layout(
        title="QHF Distributions & 3D Relationship",
        bargap=0.02,
        margin=dict(l=10, r=10, t=40, b=10),
        height=800
    )
    return fig




def build_scatterplanes_fig(data):
    """
    Build pairwise scatter plot figure showing relationships between variables.
    
    Args:
        data: QHF results data dictionary
        
    Returns:
        go.Figure: Plotly figure with three scatter subplots
    """
    dist = data.get("distributions", {}) or {}
    T = dist.get("temperature", []) or []
    D = dist.get("depth", []) or []
    S = dist.get("suitability", []) or []

    labels = (data.get("meta", {}) or {}).get("labels", {}) or {}
    TL, TU = (labels.get("temperature", {}).get("label", "Temperature"),
              labels.get("temperature", {}).get("unit", "K"))
    DL, DU = (labels.get("depth", {}).get("label", "Depth"),
              labels.get("depth", {}).get("unit", "m"))
    SL, SU = (labels.get("suitability", {}).get("label", "Suitability"),
              labels.get("suitability", {}).get("unit", "unitless"))

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Temp vs Suitability", "Depth vs Suitability", "Temp vs Depth"))

    Tx, Sx = safe_clean_xy(T, S)
    Dx, S2 = safe_clean_xy(D, S)
    Tx2, Dx2 = safe_clean_xy(T, D)

    if Tx and Sx:
        fig.add_trace(go.Scatter(x=Tx, y=Sx, mode="markers"), row=1, col=1)
    if Dx and S2:
        fig.add_trace(go.Scatter(x=Dx, y=S2, mode="markers"), row=1, col=2)
    if Tx2 and Dx2:
        fig.add_trace(go.Scatter(x=Tx2, y=Dx2, mode="markers"), row=1, col=3)

    # Add axis labels for each subplot
    fig.update_xaxes(title_text=ulabel(TL, TU), row=1, col=1)  # Temp vs Suitability: x-axis = Temperature
    fig.update_yaxes(title_text=ulabel(SL, SU), row=1, col=1)  # Temp vs Suitability: y-axis = Suitability
    
    fig.update_xaxes(title_text=ulabel(DL, DU), row=1, col=2)  # Depth vs Suitability: x-axis = Depth
    fig.update_yaxes(title_text=ulabel(SL, SU), row=1, col=2)  # Depth vs Suitability: y-axis = Suitability
    
    fig.update_xaxes(title_text=ulabel(TL, TU), row=1, col=3)  # Temp vs Depth: x-axis = Temperature
    fig.update_yaxes(title_text=ulabel(DL, DU), row=1, col=3)  # Temp vs Depth: y-axis = Depth

    fig.update_layout(title="Pairwise Relationships",
                      margin=dict(l=10, r=10, t=40, b=10), height=420)
    return fig


# ======================================
# DASH APPLICATION SETUP
# ======================================

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "QHF Dashboard"

# ======================================
# CUSTOM FLASK ROUTES
# ======================================

@app.server.route('/video/<path:filename>')
def serve_video(filename):
    """
    Serve video files from Analyses folder.
    
    This custom route allows serving videos from the Analyses directory,
    which is not in Dash's default static file serving path.
    
    Args:
        filename: Video filename from Analyses folder
        
    Returns:
        Flask Response: Video file with proper MIME type and headers
    """
    from flask import send_from_directory, abort, Response
    import mimetypes
    analyses_path = Path(__file__).parent.resolve()
    video_path = analyses_path / filename
    # Security: ensure file is in Analyses folder
    try:
        video_path.resolve().relative_to(analyses_path.resolve())
    except ValueError:
        abort(403)  # Forbidden - file outside Analyses folder
    if video_path.exists() and video_path.is_file():
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(video_path))
        if not mime_type:
            mime_type = 'video/mp4'  # Default to mp4
        # Send file with proper MIME type and headers for video streaming
        response = send_from_directory(str(analyses_path), filename, mimetype=mime_type)
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Type'] = mime_type
        return response
    abort(404)

app.layout = html.Div(
    [
        html.Div(
            [
                html.H2("QHF Interactive Dashboard", style={"color": "white"}),
                html.Div([
                    html.Button("Reload Data", id="reload-btn", n_clicks=0,
                                style={"background": "#1E90FF", "color": "white",
                                       "border": "none", "padding": "6px 12px", "borderRadius": "6px"}),
                    html.Span("  (reads results/latest.json)",
                              style={"marginLeft": "8px", "color": "#eee"}),
                ], style={"marginBottom": "10px"}),
                html.Div(id="status-line",
                         style={"marginBottom": "6px", "color": "#eee", "fontSize": "12px"}),

                dcc.Tabs(
                    id="tabs", value="tab-home",
                    children=[
                        dcc.Tab(label="Home", value="tab-home"),
                        dcc.Tab(label="Graph", value="tab-graph"),
                        dcc.Tab(label="Distributions + 3D", value="tab-dists"),
                        dcc.Tab(label="Pairwise Scatter", value="tab-scatter"),
                        dcc.Tab(label="Videos", value="tab-videos"),
                    ]
                ),
            ],
            style={"background": "rgba(0,0,0,0.5)",
                   "padding": "15px", "borderRadius": "8px",
                   "boxShadow": "0 2px 8px rgba(0,0,0,0.3)",
                   "marginBottom": "15px"}
        ),
        html.Div(id="tab-content", style={"marginTop": "10px"}),
        
        # Store for node positions and graph data
        dcc.Store(id="node-positions-store", data={}),
        dcc.Store(id="graph-data-store", data={}),
        dcc.Store(id="font-size-store", data={"size": 14}),  # Default font size
        
        # Modal for node information
        html.Div(
            id="node-info-modal",
            children=[
                html.Div(
                    [
                        html.H3(id="modal-node-name", style={"margin": "0 0 15px 0", "color": "#2c3e50"}),
                        html.Div(id="modal-node-content"),
                        html.Button(
                            "Close",
                            id="modal-close-btn",
                            n_clicks=0,
                            style={
                                "marginTop": "20px",
                                "padding": "8px 16px",
                                "background": "#1E90FF",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "4px",
                                "cursor": "pointer"
                            }
                        )
                    ],
                    style={"padding": "20px"}
                )
            ],
            style={
                "display": "none",
                "position": "fixed",
                "zIndex": "1000",
                "left": "50%",
                "top": "50%",
                "transform": "translate(-50%, -50%)",
                "width": "500px",
                "maxWidth": "90vw",
                "maxHeight": "80vh",
                "overflow": "auto",
                "background": "white",
                "borderRadius": "8px",
                "boxShadow": "0 4px 20px rgba(0,0,0,0.3)"
            }
        )
    ],
    style={"backgroundImage": "url('assets/background.jpg')",
           "backgroundSize": "cover", "backgroundAttachment": "fixed",
           "backgroundPosition": "center", "minHeight": "100vh",
           "padding": "20px"}
)


# ======================================
# DASH CALLBACKS
# ======================================

@app.callback(
    [Output("tab-content", "children"), Output("status-line", "children"),
     Output("graph-data-store", "data"), Output("node-positions-store", "data")],
    [Input("tabs", "value"), Input("reload-btn", "n_clicks")],
    [State("node-positions-store", "data"), State("font-size-store", "data")],
    prevent_initial_call=False
)
def render_tab(tab, _n, stored_positions, font_size_data):
    """
    Render content for the selected tab (Home, Graph, Distributions, Scatter, Videos).
    
    Args:
        tab: Selected tab value
        _n: Reload button click count
        stored_positions: Previously stored node positions
        font_size_data: Current font size setting
        
    Returns:
        tuple: (tab_content, status_message, graph_data, node_positions)
    """
    try:
        data = load_data()
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        return html.Div(error_msg, style={"color": "red"}), error_msg, {}, {}
    
    # Store graph data for use in other callbacks
    gdata = data.get("graph", {}) or {}
    graph_data = {
        "nodes": gdata.get("nodes", []),
        "edges": gdata.get("edges", []),
        "node_colors": gdata.get("node_colors", {}),
        "directed": gdata.get("directed", True)
    }
    
    # Use stored positions if available, otherwise use data positions
    initial_positions = gdata.get("positions", {}) or {}
    if stored_positions and isinstance(stored_positions, dict) and stored_positions.get("positions"):
        # Merge: use stored positions for nodes that exist, add new nodes from initial
        stored = stored_positions.get("positions", {})
        node_positions = {**initial_positions, **stored}
        # Only keep positions for nodes that exist
        nodes = gdata.get("nodes", [])
        node_positions = {k: v for k, v in node_positions.items() if k in nodes}
    else:
        node_positions = initial_positions

    meta = data.get("meta", {}) or {}
    diag = meta.get("diagnostics", {}) or {}
    nw = int(diag.get("n_warnings", 0))
    inv = diag.get("invalid_counts", {}) or {}
    banner = None
    if nw > 0:
        banner = html.Div(
            f"Warnings: {nw} | Invalid counts: " +
            ", ".join(f"{k}={v}" for k, v in inv.items()),
            style={"background": "#fff3cd",
                   "border": "1px solid #ffeeba",
                   "padding": "6px", "margin": "6px 0", "fontSize": "12px"}
        )

    dist = data.get("distributions", {})
    def finite_count(x): return int(np.isfinite(np.asarray(x, dtype=float)).sum())
    def rng(x):
        a = np.asarray(x, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            return "[]"
        return f"[{float(a.min()):.3g}-{float(a.max()):.3g}]"

    status = (f"Loaded: T={finite_count(dist.get('temperature', []))} "
              f"D={finite_count(dist.get('depth', []))} "
              f"S={finite_count(dist.get('suitability', []))} "
              f"| ranges: T={rng(dist.get('temperature', []))} "
              f"D={rng(dist.get('depth', []))} "
              f"S={rng(dist.get('suitability', []))} "
              f"| file: {data.get('_results_path','?')}")
              

    if tab == "tab-home":
        return html.Div([
            html.H2("Quantitative Habitability Framework (QHF)", 
                   style={"color": "white", "marginBottom": "20px", "fontSize": "28px"}),
            
            html.Div([
                html.H3("QuantHab Framework Paper in Press!", 
                       style={"color": "#4CAF50", "marginTop": "20px", "marginBottom": "15px"}),
                html.P("After a comprehensive 4 year-long effort, the QuantHab Science Working Group completed its work and the Quantitative Habitability Framework (QHF) manuscript is now in press in the Planetary Science Journal! The QHF manuscript introduces a new terminology and quantitative, probabilistic framework for assessing organism/ecosystem viability in potential habitats. The QHF framework is based on a self-consistent terminology, informed by ecological models and viability assessment, and designed to correctly incorporate incomplete and uncertain data on organisms and habitats.",
                       style={"color": "#eee", "fontSize": "15px", "lineHeight": "1.6", "marginBottom": "15px"}),
                html.P("Along with the framework, we are releasing an open-source python implementation, which also includes four different applications as examples: (a) Prioritization of TRAPPIST-1ef-like planets for biosignatures searches; (b) Assessment of Cyanobacteria viability in TRAPPIST-1ef-like planets to support interpretation of potential future O2 detection; (c) Habitability of the Martian subsurface; (d) Habitability of Europa's ocean.",
                       style={"color": "#eee", "fontSize": "15px", "lineHeight": "1.6", "marginBottom": "20px"}),
            ], style={"background": "rgba(0,0,0,0.3)", "padding": "20px", "borderRadius": "8px", "marginBottom": "20px"}),
            
            html.Div([
                html.H3("The QuantHab Science Working Group", 
                       style={"color": "#64B5F6", "marginTop": "20px", "marginBottom": "15px"}),
                html.P("The NExSS research coordination network launched its Quantitative Habitability Science Working Group (QuantHab) in August 2020. The QuantHab SWG was proposed by Daniel Apai, and the SWG was co-chaired by Daniel Apai and Rory Barnes. We kicked off the process by holding a workshop in December 2020 to engage the community in the process. Over a hundred participants joined and provided input and talks. For nearly three years, we organized biweekly meetings to further develop ideas, models, and the concept for the paper. In May 2025, the manuscript describing the completed effort was accepted in the Planetary Science Journal, which fully completes the charge of the QuantHab SWG. We are looking forward to applying QHF across a suite of astrobiology projects.",
                       style={"color": "#eee", "fontSize": "15px", "lineHeight": "1.6", "marginBottom": "20px"}),
            ], style={"background": "rgba(0,0,0,0.3)", "padding": "20px", "borderRadius": "8px", "marginBottom": "20px"}),
            
            html.Div([
                html.H3("Charge of the QuantHab Science Working Group", 
                       style={"color": "#FFB74D", "marginTop": "20px", "marginBottom": "15px"}),
                html.P("The assessment of planetary habitability is at the core of the search for life on exoplanets, but it remains a complex and poorly constrained problem. Constraints are now emerging, at an increasing pace, from observations and models of planet formation, planet evolution, stellar characterization, present-day atmospheric composition, as well as from exoplanet population statistics and specific, but necessarily incomplete and often uncertain information on the specific planet targeted. Future exoplanet characterization efforts will necessarily have to work with such incomplete information and an integrative approach will be key to correct quantitative and statistical interpretation of the potential surface habitability of given targets, also underpinning the interpretation of potential biosignatures. The multi-disciplinary exoplanet communities continue to make rapid progress on focused research, but integrating evidence – often statistical in nature – across disciplines and sub-fields remains a major challenge. NExSS is uniquely well positioned to provide a hub and conduit for such an integrative effort.",
                       style={"color": "#eee", "fontSize": "15px", "lineHeight": "1.6", "marginBottom": "15px"}),
                html.P("The QuantHab Science Working Group aims to accomplish the following goals: Establish efficient channels of communication for the relevant groups; Identify and connect to existing resources and activities to avoid duplication and maximize efficiency; Establish a centralized online hub to collect and organize relevant datasets, publications, links to groups; Organize quarterly workshops focused on integrating quantitative knowledge on habitability.",
                       style={"color": "#eee", "fontSize": "15px", "lineHeight": "1.6", "marginBottom": "20px"}),
            ], style={"background": "rgba(0,0,0,0.3)", "padding": "20px", "borderRadius": "8px", "marginBottom": "20px"}),
            
            html.Div([
                html.H3("About This Dashboard", 
                       style={"color": "#BA68C8", "marginTop": "20px", "marginBottom": "15px"}),
                html.P("This dashboard visualizes results from the Quantitative Habitability Framework (QHF). Explore planetary module dependencies, analyze variable distributions, and view relationships in 3D. Use the interactive graph to understand how different planetary modules connect and influence each other. Click on nodes to learn more about each module, and use the various visualization tools to explore your data.",
                       style={"color": "#eee", "fontSize": "15px", "lineHeight": "1.6", "marginBottom": "15px"}),
                html.P("For more information, visit: ", 
                       style={"color": "#eee", "fontSize": "15px", "display": "inline"}),
                html.A("https://alienearths.space/quanthab/", 
                      href="https://alienearths.space/quanthab/",
                      target="_blank",
                      style={"color": "#64B5F6", "fontSize": "15px", "textDecoration": "underline"}),
            ], style={"background": "rgba(0,0,0,0.3)", "padding": "20px", "borderRadius": "8px", "marginBottom": "20px"}),
            
        ], style={"maxWidth": "1000px", "margin": "0 auto"}), status, graph_data, {"positions": node_positions}

    elif tab == "tab-graph":
        # Get font size from store (with default)
        if font_size_data and isinstance(font_size_data, dict):
            font_size = font_size_data.get("size", 14)
        else:
            font_size = 14
        fig = build_graph_fig(data, node_positions=node_positions, font_size=font_size)
        toolbar = html.Div([
            html.Button("＋ Zoom In", id="zoom-in", n_clicks=0, style={"marginRight": "6px"}),
            html.Button("－ Zoom Out", id="zoom-out", n_clicks=0, style={"marginRight": "6px"}),
            html.Button("⟲ Reset", id="zoom-reset", n_clicks=0, style={"marginRight": "6px"}),
            html.Span("  |  ", style={"marginLeft": "6px", "marginRight": "6px", "color": "#eee"}),
            html.Button("A+", id="font-increase", n_clicks=0, 
                       title="Increase font size",
                       style={"marginRight": "4px", "padding": "4px 8px", "fontSize": "12px"}),
            html.Button("A-", id="font-decrease", n_clicks=0,
                       title="Decrease font size",
                       style={"marginRight": "6px", "padding": "4px 8px", "fontSize": "12px"}),
            html.Span("Font Size", 
                     style={"marginRight": "6px", "color": "#eee", "fontSize": "12px"}),
            html.Span(id="font-size-display", children="14",
                     style={"marginRight": "12px", "color": "#fff", "fontSize": "12px", "fontWeight": "bold"}),
            html.Span("  |  Click nodes for info", 
                     style={"marginLeft": "6px", "color": "#eee", "fontSize": "12px"})
        ], style={"marginBottom": "8px"})
        graph = dcc.Graph(
            id="graph-plot", 
            figure=fig, 
            style={"height": "70vh"},
            config={
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "zoom2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"
                    # select2d and lasso2d buttons are available by default in the mode bar
                ],
                "editable": False,
                "displayModeBar": True,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "qhf_graph",
                    "height": 800,
                    "width": 1200,
                    "scale": 2
                },
                # Enable interactive features
                "doubleClick": "reset",
                "showTips": True
            }
        )
        children = [toolbar, graph]
        if banner:
            children = [banner] + children
        return children, status, graph_data, {"positions": node_positions}

    elif tab == "tab-dists":
        fig = build_distributions_fig(data)
        return [dcc.Graph(figure=fig, style={"height": "85vh"})], status, graph_data, {"positions": node_positions}

    elif tab == "tab-scatter":
        fig = build_scatterplanes_fig(data)
        return [dcc.Graph(figure=fig, style={"height": "70vh"})], status, graph_data, {"positions": node_positions}

    elif tab == "tab-videos":
        # Find video file in assets folder (Dash serves static files from assets/)
        # Also check Analyses folder and suggest moving to assets if found there
        analyses_path = Path(__file__).parent.resolve()
        assets_path = analyses_path / "assets"
        video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
        video_file = None
        video_url = None
        video_in_analyses = False
        
        # First check assets folder (preferred - Dash serves from /assets/)
        if assets_path.exists():
            for file_path in assets_path.iterdir():
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in [e.lower() for e in video_extensions]:
                        video_file = file_path.name
                        video_url = f"/assets/{video_file}"  # Dash serves from /assets/
                        break
        
        # If not in assets, check Analyses folder
        if video_file is None and analyses_path.exists():
            for file_path in analyses_path.iterdir():
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in [e.lower() for e in video_extensions]:
                        video_file = file_path.name
                        video_in_analyses = True
                        # Check if same file exists in assets (case-insensitive)
                        if assets_path.exists():
                            for asset_file in assets_path.iterdir():
                                if asset_file.is_file() and asset_file.suffix.lower() == ext:
                                    video_file = asset_file.name
                                    video_url = f"/assets/{video_file}"
                                    video_in_analyses = False
                                    break
                        # If still in Analyses, serve via custom route
                        if video_in_analyses:
                            video_url = f"/video/{video_file}"  # Use custom route - Flask handles encoding
                        break
        
        # Build video content
        video_content = []
        
        # If video found in Analyses but not in assets, show message (but video_url should be set now)
        if video_file and video_in_analyses and not video_url:
            video_content = [
                html.Div([
                    html.Div([
                        html.Span("⚠️", style={"fontSize": "48px", "display": "block", "marginBottom": "15px"}),
                        html.H3("Video Found - Action Required",
                               style={"color": "#ffd700", "fontSize": "22px", "marginBottom": "15px", "fontWeight": "600"}),
                        html.P(f"Found video file: {video_file}",
                               style={"color": "#ddd", "fontSize": "16px", "lineHeight": "1.8", "marginBottom": "20px"}),
                        html.Div([
                            html.P("📁 To display the video:", style={"color": "#aaa", "fontSize": "14px", "fontWeight": "600", "marginBottom": "8px"}),
                            html.P(f"Move '{video_file}' from the Analyses/ folder to the Analyses/assets/ folder, then refresh this page.",
                                   style={"color": "#ffd700", "fontSize": "15px", "lineHeight": "1.6", "fontWeight": "500"})
                        ], style={"background": "rgba(255,215,0,0.1)", "padding": "20px", "borderRadius": "8px", "marginTop": "15px", "border": "1px solid rgba(255,215,0,0.3)"})
                    ], style={"textAlign": "center"})
                ], style={
                    "background": "rgba(0,0,0,0.4)",
                    "padding": "50px 40px",
                    "borderRadius": "12px",
                    "boxShadow": "0 2px 10px rgba(0,0,0,0.3)"
                })
            ]
        elif video_file and video_url:
            video_content = [
                html.Div([
                    html.H3("QHF Video Tutorial", 
                           style={"color": "white", "marginBottom": "20px", "fontSize": "26px", "textAlign": "center", "fontWeight": "600"}),
                    html.Div([
                        html.Video(
                            id="qhf-video",
                            src=video_url,
                            controls=True,
                            autoPlay=False,
                            style={
                                "width": "100%",
                                "maxWidth": "1000px",
                                "height": "auto",
                                "borderRadius": "12px",
                                "boxShadow": "0 6px 25px rgba(0,0,0,0.6)",
                                "backgroundColor": "#000",
                                "outline": "none"
                            }
                        )
                    ], style={"display": "flex", "justifyContent": "center", "marginBottom": "25px"}),
                    html.Div([
                        html.P(f"📹 {video_file}",
                               style={"color": "#ccc", "fontSize": "15px", "textAlign": "center", "marginBottom": "5px"}),
                        html.P("Use the controls above to play, pause, and adjust volume.",
                               style={"color": "#888", "fontSize": "13px", "textAlign": "center", "fontStyle": "italic"})
                    ])
                ], style={
                    "background": "linear-gradient(135deg, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0.3) 100%)",
                    "padding": "50px 40px",
                    "borderRadius": "16px",
                    "boxShadow": "0 4px 15px rgba(0,0,0,0.4)",
                    "border": "1px solid rgba(255,255,255,0.1)"
                })
            ]
        else:
            # No video found - show helpful message with debug info
            # Get list of all files for debugging
            analyses_files = []
            assets_files = []
            if analyses_path.exists():
                analyses_files = [f.name for f in analyses_path.iterdir() if f.is_file()]
            if assets_path.exists():
                assets_files = [f.name for f in assets_path.iterdir() if f.is_file()]
            
            debug_info = []
            if analyses_files:
                debug_info.append(html.P(f"Files in Analyses/: {', '.join(analyses_files[:10])}" + ("..." if len(analyses_files) > 10 else ""),
                                       style={"color": "#888", "fontSize": "12px", "fontFamily": "monospace", "marginTop": "10px"}))
            if assets_files:
                debug_info.append(html.P(f"Files in assets/: {', '.join(assets_files[:10])}" + ("..." if len(assets_files) > 10 else ""),
                                       style={"color": "#888", "fontSize": "12px", "fontFamily": "monospace", "marginTop": "5px"}))
            
            video_content = [
                html.Div([
                    html.Div([
                        html.Span("🎬", style={"fontSize": "48px", "display": "block", "marginBottom": "15px"}),
                        html.H3("Video Content Coming Soon",
                               style={"color": "white", "fontSize": "22px", "marginBottom": "15px", "fontWeight": "600"}),
                        html.P("Video content will be displayed here for educational videos, tutorials, and demonstrations related to the Quantitative Habitability Framework.",
                               style={"color": "#ddd", "fontSize": "16px", "lineHeight": "1.8", "marginBottom": "20px"}),
                        html.Div([
                            html.P("📁 To add a video:", style={"color": "#aaa", "fontSize": "14px", "fontWeight": "600", "marginBottom": "8px"}),
                            html.Ul([
                                html.Li("Place your video file (MP4, AVI, MOV, WebM, or MKV) in the Analyses/assets/ folder", 
                                       style={"color": "#bbb", "fontSize": "14px", "marginBottom": "5px"}),
                                html.Li("Or place it directly in the Analyses/ folder (will prompt you to move it)", 
                                       style={"color": "#bbb", "fontSize": "14px", "marginBottom": "5px"}),
                                html.Li("Supported names: video.mp4, QHF.mp4, tutorial.mp4, or any video file", 
                                       style={"color": "#bbb", "fontSize": "14px"})
                            ], style={"textAlign": "left", "display": "inline-block", "marginTop": "10px"})
                        ] + debug_info, style={"background": "rgba(0,0,0,0.2)", "padding": "20px", "borderRadius": "8px", "marginTop": "15px"})
                    ], style={"textAlign": "center"})
                ], style={
                    "background": "rgba(0,0,0,0.3)",
                    "padding": "50px 40px",
                    "borderRadius": "12px",
                    "boxShadow": "0 2px 10px rgba(0,0,0,0.3)"
                })
            ]
        
        return html.Div([
            html.H2("Videos", style={
                "color": "white", 
                "marginBottom": "30px", 
                "fontSize": "36px", 
                "textAlign": "center", 
                "fontWeight": "bold",
                "textShadow": "2px 2px 4px rgba(0,0,0,0.5)"
            }),
            html.Div(video_content, style={"maxWidth": "1100px", "margin": "0 auto"})
        ], style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"}), status, graph_data, {"positions": node_positions}

    else:
        return [html.Div("Unknown tab", style={"color": "white"})], status, graph_data, {"positions": node_positions}


# ======================================
# GRAPH INTERACTION CALLBACKS
# ======================================

@app.callback(
    Output("graph-plot", "figure"),
    [Input("zoom-in", "n_clicks"),
     Input("zoom-out", "n_clicks"),
     Input("zoom-reset", "n_clicks"),
     Input("graph-plot", "relayoutData")],
    [State("graph-plot", "figure"),
     State("node-positions-store", "data"),
     State("graph-data-store", "data")],
    prevent_initial_call=True
)
def on_zoom(zin, zout, zreset, relayout_data, fig, stored_positions, graph_data):
    """
    Handle zoom in, zoom out, and reset button clicks for the graph.
    
    Args:
        zin: Zoom in button clicks
        zout: Zoom out button clicks
        zreset: Reset button clicks
        relayout_data: Plotly relayout event data
        fig: Current figure state
        stored_positions: Stored node positions
        graph_data: Graph data dictionary
        
    Returns:
        go.Figure: Updated figure with new zoom level
    """
    if not callback_context.triggered:
        return fig
    
    trigger = callback_context.triggered[0]["prop_id"]
    
    # Handle zoom buttons
    if "zoom-in" in trigger:
        return _apply_zoom(fig, factor=0.8)
    if "zoom-out" in trigger:
        return _apply_zoom(fig, factor=1.25)
    if "zoom-reset" in trigger:
        return _apply_zoom(fig, reset=True)
    
    # Handle node dragging via relayoutData
    if relayout_data and "graph-plot.relayoutData" in trigger:
        # Check if this is a node drag (points were moved)
        if fig and "data" in fig:
            # Try to extract new positions from relayoutData
            # Plotly doesn't directly expose point positions in relayoutData for scatter plots
            # We'll need to use a different approach - using selectedData or custom JS
            pass
    
    return fig


@app.callback(
    [Output("node-info-modal", "style"),
     Output("modal-node-name", "children"),
     Output("modal-node-content", "children")],
    [Input("graph-plot", "clickData"),
     Input("modal-close-btn", "n_clicks")],
    [State("graph-data-store", "data")],
    prevent_initial_call=True
)
def show_node_info(click_data, close_clicks, graph_data):
    """
    Display node information modal when a node is clicked.
    
    Args:
        click_data: Plotly click event data
        close_clicks: Close button click count
        graph_data: Graph data dictionary with nodes and edges
        
    Returns:
        tuple: (modal_style, node_name, modal_content)
    """
    if not callback_context.triggered:
        return {"display": "none"}, "", ""
    
    trigger = callback_context.triggered[0]["prop_id"]
    
    # Close modal
    if "modal-close-btn" in trigger:
        return {"display": "none"}, "", ""
    
    # Show node info
    if click_data and "graph-plot.clickData" in trigger:
        try:
            point_data = click_data.get("points", [{}])[0]
            node_name = point_data.get("customdata")
            if not node_name:
                return {"display": "none"}, "", ""
            
            # Build node information
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])
            
            # Find incoming and outgoing edges
            incoming = [(e[0], e[2]) for e in edges if e[1] == node_name]
            outgoing = [(e[1], e[2]) for e in edges if e[0] == node_name]
            
            content = []
            
            # Node description
            content.append(html.P(
                f"This module represents: {node_name.replace(chr(10), ' ').strip()}",
                style={"marginBottom": "15px", "fontSize": "14px", "color": "#555"}
            ))
            
            # Incoming connections
            if incoming:
                content.append(html.H4("Receives from:", style={"marginTop": "15px", "marginBottom": "8px", "fontSize": "16px", "color": "#2c3e50"}))
                in_list = html.Ul([
                    html.Li(f"{source} → {label}", style={"marginBottom": "5px", "fontSize": "13px"})
                    for source, label in incoming
                ])
                content.append(in_list)
            
            # Outgoing connections
            if outgoing:
                content.append(html.H4("Sends to:", style={"marginTop": "15px", "marginBottom": "8px", "fontSize": "16px", "color": "#2c3e50"}))
                out_list = html.Ul([
                    html.Li(f"{label} → {target}", style={"marginBottom": "5px", "fontSize": "13px"})
                    for target, label in outgoing
                ])
                content.append(out_list)
            
            if not incoming and not outgoing:
                content.append(html.P("No connections found for this node.", style={"fontSize": "13px", "color": "#888"}))
            
            modal_style = {
                "display": "block",
                "position": "fixed",
                "zIndex": "1000",
                "left": "50%",
                "top": "50%",
                "transform": "translate(-50%, -50%)",
                "width": "500px",
                "maxWidth": "90vw",
                "maxHeight": "80vh",
                "overflow": "auto",
                "background": "white",
                "borderRadius": "8px",
                "boxShadow": "0 4px 20px rgba(0,0,0,0.3)"
            }
            
            return modal_style, node_name.replace("\n", " ").strip(), content
            
        except Exception as e:
            return {"display": "none"}, "", ""
    
    return {"display": "none"}, "", ""


@app.callback(
    [Output("node-positions-store", "data", allow_duplicate=True),
     Output("graph-plot", "figure", allow_duplicate=True)],
    [Input("graph-plot", "figure")],
    [State("node-positions-store", "data"),
     State("graph-data-store", "data"),
     State("font-size-store", "data")],
    prevent_initial_call=True
)
def sync_node_positions(fig, stored_positions, graph_data, font_size_data):
    """
    Synchronize node positions from the figure to the store.
    
    This callback extracts current node positions from the figure trace
    and updates the stored positions. It also rebuilds edges and annotations
    based on the new positions.
    
    Args:
        fig: Current Plotly figure
        stored_positions: Previously stored positions
        graph_data: Graph data dictionary
        font_size_data: Current font size setting
        
    Returns:
        tuple: (updated_positions_dict, updated_figure)
    """
    if not fig or not fig.get("data"):
        return stored_positions, fig
    
    # Get current font size
    font_size = font_size_data.get("size", 14) if font_size_data else 14
    
    # Find the node trace
    nodes = graph_data.get("nodes", []) if graph_data else []
    if not nodes:
        return stored_positions, fig
    
    node_trace = None
    for trace in fig.get("data", []):
        if trace.get("mode") == "markers" or (isinstance(trace.get("mode"), str) and "markers" in trace.get("mode")):
            node_trace = trace
            break
    
    if not node_trace or not node_trace.get("customdata"):
        return stored_positions, fig
    
    # Extract positions from the CURRENT trace data (which may have been updated by JavaScript)
    node_x = node_trace.get("x", [])
    node_y = node_trace.get("y", [])
    node_names = node_trace.get("customdata", [])
    
    if len(node_x) != len(node_y) or len(node_x) != len(node_names):
        return stored_positions, fig
    
    # Update stored positions from CURRENT trace data (preserve JavaScript updates)
    new_positions = {}
    for i, node_name in enumerate(node_names):
        if i < len(node_x) and i < len(node_y):
            # Always use the current trace positions - these reflect any JavaScript updates
            new_positions[node_name] = [float(node_x[i]), float(node_y[i])]
    
    # Rebuild edges based on new positions
    edges = graph_data.get("edges", []) if graph_data else []
    edge_x, edge_y = [], []
    for (u, v, _lab) in edges:
        if u in new_positions and v in new_positions:
            x0, y0 = new_positions[u]
            x1, y1 = new_positions[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
    
    # Update edge trace
    edge_trace = None
    for trace in fig.get("data", []):
        if trace.get("mode") == "lines":
            edge_trace = trace
            break
    
    if edge_trace:
        edge_trace["x"] = edge_x
        edge_trace["y"] = edge_y
        # Update edge styling to match new interactive style
        if "line" not in edge_trace:
            edge_trace["line"] = {}
        edge_trace["line"]["width"] = 2
        edge_trace["line"]["color"] = "rgba(100,149,237,0.6)"
    
    # Update node trace text and styling if needed
    if node_trace:
        # Ensure text is set on nodes
        if "text" not in node_trace or not node_trace.get("text"):
            node_trace["text"] = [n.replace("\n", " ") for n in node_trace.get("customdata", [])]
            node_trace["textposition"] = "middle center"
            if "textfont" not in node_trace:
                node_trace["textfont"] = {}
            node_trace["textfont"]["size"] = max(10, int(font_size * 0.7))
            node_trace["textfont"]["color"] = "black"
            node_trace["textfont"]["family"] = "Arial Black"
        # Update marker size for better visibility
        if "marker" not in node_trace:
            node_trace["marker"] = {}
        if "size" not in node_trace["marker"]:
            node_trace["marker"]["size"] = 25
        if "line" not in node_trace["marker"]:
            node_trace["marker"]["line"] = {}
        node_trace["marker"]["line"]["width"] = 2
        node_trace["marker"]["line"]["color"] = "#1a1a1a"
    
    # Update arrow annotations with labels
    if graph_data and graph_data.get("directed", True) and edges:
        anns = []
        for (u, v, edge_label) in edges:
            if u in new_positions and v in new_positions:
                x0, y0 = new_positions[u]
                x1, y1 = new_positions[v]
                xm, ym = (x0 + x1)/2, (y0 + y1)/2
                dx, dy = (x1 - x0), (y1 - y0)
                
                # Position label on the line (no perpendicular offset)
                # Create arrow annotation with label positioned on the line
                ann = dict(
                    x=xm, y=ym,  # Position label on the line center
                    ax=xm - 0.1*dx, ay=ym - 0.1*dy,  # Arrow positioned closer to source
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
                    arrowcolor="rgba(100,149,237,0.8)",
                    text=str(edge_label) if edge_label else "",  # Add edge label text
                    xanchor="center", yanchor="middle",
                    font=dict(size=max(10, int(font_size * 0.75)), color="#1a1a1a", family="Arial"),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="rgba(100,149,237,0.8)", borderwidth=1.5, borderpad=4
                )
                anns.append(ann)
        
        # Keep non-arrow annotations (planet name) and add updated arrows
        # Node labels are now on nodes themselves, not as separate annotations
        existing_anns = [ann for ann in (fig.get("layout", {}).get("annotations", []) or []) 
                        if not ann.get("showarrow")]
        fig["layout"]["annotations"] = existing_anns + anns
    
    # Update node positions from trace (they may have been updated by JavaScript)
    if node_trace:
        node_x_current = node_trace.get("x", [])
        node_y_current = node_trace.get("y", [])
        node_names_current = node_trace.get("customdata", [])
        
        # Update positions from current trace data
        for i, node_name in enumerate(node_names_current):
            if i < len(node_x_current) and i < len(node_y_current):
                new_positions[node_name] = [float(node_x_current[i]), float(node_y_current[i])]
    
    # Node labels are now displayed directly on nodes via text in Scatter trace
    # No need to update separate label annotations
    # Just preserve existing annotations (arrows and planet name)
    existing_anns = [ann for ann in (fig.get("layout", {}).get("annotations", []) or []) 
                     if ann.get("showarrow") or (ann.get("xref") == "paper" and not ann.get("showarrow"))]
    fig["layout"]["annotations"] = existing_anns
    
    return {"positions": new_positions}, fig


@app.callback(
    [Output("font-size-store", "data"),
     Output("font-size-display", "children"),
     Output("graph-plot", "figure", allow_duplicate=True)],
    [Input("font-increase", "n_clicks"),
     Input("font-decrease", "n_clicks")],
    [State("font-size-store", "data"),
     State("graph-plot", "figure"),
     State("graph-data-store", "data"),
     State("node-positions-store", "data")],
    prevent_initial_call=True
)
def update_font_size(inc_clicks, dec_clicks, font_size_data, fig, graph_data, stored_positions):
    """
    Handle font size increase/decrease and rebuild graph with new font size.
    
    Args:
        inc_clicks: Font increase button clicks
        dec_clicks: Font decrease button clicks
        font_size_data: Current font size setting
        fig: Current figure
        graph_data: Graph data dictionary
        stored_positions: Stored node positions
        
    Returns:
        tuple: (new_font_size_data, font_size_display_text, updated_figure)
    """
    if not callback_context.triggered:
        return font_size_data, str(font_size_data.get("size", 14) if font_size_data else 14), fig
    
    trigger = callback_context.triggered[0]["prop_id"]
    current_size = font_size_data.get("size", 14) if font_size_data else 14
    
    # Update font size
    if "font-increase" in trigger:
        new_size = min(24, current_size + 2)  # Max 24, increase by 2
    elif "font-decrease" in trigger:
        new_size = max(8, current_size - 2)  # Min 8, decrease by 2
    else:
        new_size = current_size
    
    new_font_data = {"size": new_size}
    
    # Rebuild the graph with new font size
    if fig and graph_data:
        # Get current node positions
        node_positions = stored_positions.get("positions", {}) if stored_positions else {}
        
        # Rebuild figure with new font size
        # We need to load the data again to rebuild
        try:
            data = load_data()
            new_fig = build_graph_fig(data, node_positions=node_positions, font_size=new_size)
            return new_font_data, str(new_size), new_fig
        except Exception:
            return new_font_data, str(new_size), fig
    
    return new_font_data, str(new_size), fig


# ======================================
# APPLICATION ENTRY POINT
# ======================================

if __name__ == "__main__":
    app.run(debug=True)
