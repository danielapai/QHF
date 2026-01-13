"""
QHF Main Execution Script

This is the main entry point for running the Quantitative Habitability Framework (QHF).
It loads configuration files, dynamically imports habitat and metabolism modules,
builds a dependency graph, performs Monte Carlo simulations, and generates visualizations.

Program Flow:
1. Load configuration file
2. Dynamically import habitat and metabolism modules
3. Build dependency graph from module connections
4. Perform topological sorting
5. Execute Monte Carlo simulation
6. Export results for dashboard
7. Generate visualizations

Author: QHF Development Team
"""

# ======================================
# IMPORTS
# ======================================

import sys
import os
sys.path.append('./Habitats')
sys.path.append('./Metabolisms')
sys.path.append('./Analyses')

from collections import defaultdict
from Utils.dashboard_io import write_results_for_dashboard
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.patches as patches
import importlib
import configparser  
import matplotlib.patheffects as pe
import pdb           
import keyparams
from mcmodules import Module as Module
from layout_presets import presets, label_offsets
import importlib.util
from matplotlib import colors as _mcolors


# ======================================
# VALIDATION CONFIGURATION
# ======================================

# Validation ranges for output parameters
# These ranges are used to detect invalid values during simulation
RANGES = {
    "Temperature":        (0.0, 1000.0),
    "Pressure":           (0.0, None),
    "Bond_Albedo":        (0.0, 1.0),
    "GreenhouseWarming":  (0.0, None),
    "Suitability":        (0.0, 1.0),
    "Depth":              (None, None),  # numeric, any real (space/time axis OK)
}

def _f(x):
    """
    Convert value to float, handling None and invalid values.
    
    Args:
        x: Input value to convert
        
    Returns:
        float: Converted value, or NaN if invalid
    """
    try:
        if x is None:
            return float("nan")
        result = float(x)
        # Check for NaN or infinite values
        if not np.isfinite(result):
            return float("nan")
        return result
    except Exception:
        return float("nan")

def check_range(name, val, lo, hi, context=""):
    """
    Validate a value against expected range and record diagnostics.
    
    Args:
        name: Parameter name
        val: Value to check
        lo: Lower bound (None for no limit)
        hi: Upper bound (None for no limit)
        context: Additional context string for error messages
        
    Returns:
        float: Validated value (may be NaN if invalid)
    """
    v = _f(val)
    bad = (not np.isfinite(v)) or (lo is not None and v < lo) or (hi is not None and v > hi)
    if bad:
        diagnostics["invalid_counts"][name] += 1
        diagnostics["warnings"].append(f"{name}={val} out of range {lo}-{hi} {context}")
    return v

def safe_clean_value(val, fallback=0.0):
    """
    Clean a value, replacing NaN/inf with fallback.
    
    Args:
        val: Value to clean
        fallback: Default value if val is invalid
        
    Returns:
        float: Cleaned value
    """
    try:
        if val is None:
            return fallback
        result = float(val)
        if not np.isfinite(result):
            return fallback
        return result
    except Exception:
        return fallback


# ======================================
# GRAPH VISUALIZATION CLASS
# ======================================

class GraphVisualization:
    """
    Class for building and visualizing module dependency graphs.
    
    This class manages the graph structure and generates matplotlib
    visualizations of module connections with NetworkX.
    """
    
    def __init__(self):
        """Initialize an empty graph visualization."""
        self.visual = []

    def addEdge(self, a, b, label):
        """
        Add an edge to the graph.
        
        Args:
            a: Source node index
            b: Target node index
            label: Edge label (parameter name)
        """
        self.visual.append([a, b])

    def visualize(self):
        """
        Generate matplotlib visualization of the dependency graph.
        
        Returns:
            matplotlib.axes.Axes: Axes object with the graph visualization
        """
        G = nx.DiGraph()
        G.add_edges_from(self.visual)

        # Choose layout spacing and pull offsets for current config
        preset_name = HabitatShortName.lower()
        offset_dict = presets.get(preset_name, {})
        label_dict = label_offsets.get(preset_name, {})

        # Spread out graph layout
        pos = nx.spring_layout(G, seed=42, k=0.7, scale=3.0, iterations=150)

        for node, label in mod_labels.items():
            normalized_label = label.replace('\n', ' ').strip()
            x, y = pos[node]
            dx, dy = offset_dict.get(label, (0.00, 0.00))
            pos[node] = (x + dx, y + dy)

        # Node color logic
        node_colors = [
            prior_node_color if len(Modules[node].input_parameters) == 0
            else metabolism_node_color if 'Suitability' in Modules[node].output_parameters
            else other_node_color
            for node in G
        ]

        node_size_val = 12 * sf  # Unified box size

        # Draw background layers
        if screen:
            nx.draw_networkx(
                G, pos, arrows=False, arrowsize=3.0 * sf, with_labels=False,
                width=3 * sf, alpha=0.02, edge_color=selected_edgecolor,
                node_color="white", node_size=70 * sf
            )
            nx.draw_networkx(
                G, pos, arrows=False, arrowsize=3.0 * sf, with_labels=False,
                width=2 * sf, alpha=0.05, edge_color=selected_edgecolor,
                node_color=node_colors, node_size=50 * sf
            )

        # Main network draw
        nx.draw_networkx(
            G, pos, arrows=True, arrowsize=3.0 * sf, with_labels=False,
            width=0.5 * sf, alpha=0.7, edge_color=selected_edgecolor,
            node_color=node_colors, node_size=node_size_val
        )

        # Edge label cleanup
        for idx, varlabel_key in enumerate(edge_labels):
            if edge_labels[varlabel_key] == 'Surface Temperature':
                edge_labels[varlabel_key] = 'Temperature'
            elif edge_labels[varlabel_key] == 'Surface Pressure':
                edge_labels[varlabel_key] = 'Pressure'

        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, label_pos=0.4, rotate=False,
            font_color=selected_edgecolor, font_size=1.6 * sf,
            font_weight='light', bbox=dict(alpha=0.2, fc=bkgcolor, ec=labelcolor, linewidth=0.1 * sf),
            clip_on=True
        )

        # Label placement
        pos_upper = {}
        for k, v in pos.items():
            pos_upper[k] = (v[0], v[1] + 0.05)

        nx.draw_networkx_labels(
            G, pos_upper, mod_labels,
            font_size=1.8 * sf,
            font_color=labelcolor,
            font_weight='light',
            horizontalalignment='center',
            verticalalignment="bottom"
        )
        
        plt.title(
            'Connections between Modules', color=labelcolor, fontsize=5,
            bbox=dict(alpha=0.1, fc=bkgcolor, ec=bkgcolor, linewidth=0.)
        )

        plt.axis("off")
        ax = plt.gca()

        if screen:
            rect = patches.Rectangle(
                (0., 0.), 1., 1., linewidth=0.2, edgecolor='lightblue',
                facecolor='none', transform=ax.transAxes
            )
            ax.add_patch(rect)

        return ax


# ======================================
# CONFIGURATION FILE LOADING
# ======================================

# Load configuration file from command line argument
cl_args = sys.argv
config_file_path = str(cl_args[1])
config = configparser.ConfigParser()
config.read(config_file_path)

ConfigID = config['Configuration']['ConfigID']
HabitatFile = config['Habitat']['HabitatFile']
HabitatModule = config['Habitat']['HabitatModule']
# Store logo path as provided (relative); resolve later in one place
HabitatLogo = config['Habitat'].get('HabitatLogo', config['Habitat'].get('Habitatlogo', ''))
HabitatShortName = config['Habitat']['HabitatShortname']

MetabolismFile = os.path.splitext(config['Metabolism']['MetabolismFile'])[0]
MetabolismModule = config['Metabolism']['MetabolismModule']

VisualizationFile = os.path.splitext(config['Visualization']['VisualizationFile'])[0]
VisualizationModule = config['Visualization']['VisualizationModule']

NumProbes = config['Sampling']['NumProbes']
if float(NumProbes) > 1e8:
    print('### Warning: Number of Probes limited -- change QHF code if you need more probes.')
NumProbes = np.clip(float(NumProbes), 1, 1e8)

print(' [ Configuration file: ]', ConfigID)
print(' [ Habitat Module: ]', HabitatModule)
print(' [ Metabolism Module: ]', MetabolismModule)
print(' [ Visualization Module: ]', VisualizationModule)


# ======================================
# DYNAMIC MODULE IMPORT
# ======================================

def dynamic_import(module_path, module_name):
    """
    Dynamically import a Python module from a file path.
    
    Args:
        module_path: Path to the module file
        module_name: Name for the imported module
        
    Returns:
        module: Imported module object
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ======================================
# MODULE LOADING
# ======================================

# Habitat module
habitat_path = os.path.join(os.path.dirname(__file__), "Habitats", HabitatFile + ".py")
habitat_module = dynamic_import(habitat_path, HabitatModule)

def _resolve_callable(module_obj, preferred_name, file_stem):
    """
    Resolve the callable function that builds modules from a module object.
    
    Handles various naming conventions and attempts to find the correct
    factory function that returns a list of modules.
    
    Args:
        module_obj: Imported module object
        preferred_name: Preferred function name from config
        file_stem: File stem for fallback naming
        
    Returns:
        callable: Function that returns a list of modules
        
    Raises:
        AttributeError: If no suitable callable is found
    """
    candidates = []
    if preferred_name:
        candidates.append(preferred_name)
        if preferred_name.endswith("Module"):
            candidates.append(preferred_name[:-6])
    # add guesses based on file name
    if file_stem:
        candidates.extend([
            file_stem,
            f"{file_stem}Modules",
            file_stem.capitalize(),
            f"{file_stem.capitalize()}Modules",
        ])
    # scan module for reasonable factory names
    for name in list(module_obj.__dict__.keys()):
        lname = str(name).lower()
        if ("modules" in lname) and callable(getattr(module_obj, name)):
            candidates.append(name)
    # try unique candidates in order
    seen = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        obj = getattr(module_obj, name, None)
        if callable(obj):
            return obj
    raise AttributeError(f"Could not resolve callable for '{preferred_name}' in module '{file_stem}'")

_habitat_factory = _resolve_callable(habitat_module, HabitatModule, HabitatFile)
Modules = _habitat_factory()

# Metabolism module
metabolism_path = os.path.join(os.path.dirname(__file__), "Metabolisms", MetabolismFile + ".py")
metabolism_module = dynamic_import(metabolism_path, MetabolismModule)
ModuleHabitability = getattr(metabolism_module, MetabolismModule)()
Modules.append(ModuleHabitability)

nmods = len(Modules)

# Visualization module
visual_path = os.path.join(os.path.dirname(__file__), "Analyses", VisualizationFile + ".py")
visual_module = dynamic_import(visual_path, VisualizationModule)
try:
    VisualizationModule = _resolve_callable(visual_module, VisualizationModule, VisualizationFile)
except Exception:
    # last resort: try a common entrypoint name; if still missing, use no-op
    VisualizationModule = getattr(visual_module, "QHFvisualize", None)
    if VisualizationModule is None:
        def _noop_visualize(*_args, **_kwargs):
            print("[INFO] No compatible matplotlib visualization function found; skipping visualization.")
        VisualizationModule = _noop_visualize

print('[Modules Loaded]')
for mi in np.arange(nmods):
    print(mi, ' : ', Modules[mi].name)


# ======================================
# VISUALIZATION THEME CONFIGURATION
# ======================================

# Theme selection: True for dark theme, False for light theme
screen = False

if screen:
    sf = 1.0
    bkgcolor = '#030810'
    selected_edgecolor = 'white'
    prior_node_color = 'blue'
    other_node_color = 'lightblue'
    metabolism_node_color = 'green'
    labelcolor = 'lightblue'
    labeloffset = 0.0
else:
    sf = 1.3
    bkgcolor = 'white'
    selected_edgecolor = 'darkblue'
    prior_node_color = 'red'
    other_node_color = 'blue'
    metabolism_node_color = 'green'
    labelcolor = 'black'
    labeloffset = -0.05


# ======================================
# DEPENDENCY GRAPH CONSTRUCTION
# ======================================

# Initialize graph visualization and label dictionaries
G = GraphVisualization()
edge_labels = {}
mod_labels = {}

for jj in np.arange(nmods):
    mod_labels[int(jj)] = Modules[jj].name
    print('--------------------------------------------------------------------------------')
    print('Identifying input connections for module ', Modules[jj].name)

    for ip in Modules[jj].input_parameters:
        print('......................................................................')
        print('Scanning for output parameters matching the input parameter:', ip)

        for module_scanned in np.arange(nmods):
            if any(x == ip for x in Modules[module_scanned].output_parameters):
                print(' + Input/output Match found in module: ', Modules[module_scanned].name)
                G.addEdge(module_scanned, jj, label=ip.replace('_', ' '))
                edge_labels[(module_scanned, jj)] = ip.replace('_', ' ')


# ======================================
# TOPOLOGICAL SORTING
# ======================================

# Perform topological sort to determine module execution order
# This ensures modules are executed after their dependencies
print("The Topological Sort Of The Graph Is: ")
DG = nx.DiGraph()
DG.add_edges_from(G.visual)
topsorted = list(nx.topological_sort(DG))
print(topsorted)


# ======================================
# MONTE CARLO SIMULATION
# ======================================

N_iter = int(config['Sampling']['Niterations'])

Suitability_Distribution = []
Temperature_Distribution = []
Pressure_Distribution = []
BondAlbedo_Distribution = []
GreenHouse_Distribution = []
Depth_Distribution = []
SavedParameters = []

N_probes = 100
Suitability_Plot = []
Variable = []

Suitability_Plot = []
Variable = []

# --- probe coordinates (depth/time/etc.) ---
NumProbes_int = int(NumProbes)
try:
    depth_min = float(config.get('Probes', 'DepthMin', fallback='0'))
    depth_max = float(config.get('Probes', 'DepthMax', fallback=str(NumProbes_int - 1)))
except Exception:
    depth_min, depth_max = 0.0, float(NumProbes_int - 1)

depth_coords = np.linspace(depth_min, depth_max, NumProbes_int)



for keyparams.ProbeIndex in np.arange(float(NumProbes)):
    print('Probing location ', keyparams.ProbeIndex)

    pi = int(keyparams.ProbeIndex)
    probe_coord = float(depth_coords[pi])
    keyparams.Depth = probe_coord          # keep for backward compatibility
    keyparams.ProbeCoord = probe_coord     # explicit, in case modules mutate Depth


    for ii in np.arange(N_iter):
        keyparams.runid = ''
        # ---- per-run diagnostics (reset each run) ----
        diagnostics = {
            "warnings": [],
            "invalid_counts": {k: 0 for k in RANGES.keys()},
        }

        for mi in np.arange(len(topsorted)):
            print('Executing ', Modules[topsorted[mi]].name)
            Modules[topsorted[mi]].execute()

        # --- robust numeric getters ---
        def _as_float(x, fallback=np.nan):
            try:
                if x is None:
                    return fallback
                return float(x)
            except Exception:
                return fallback

        # Prefer Temperature; fall back to Surface_Temperature or Equilibrium_Temp if missing
        T_raw = keyparams.Temperature
        if not np.isfinite(_as_float(T_raw, np.nan)):
            for cand in (getattr(keyparams, "Surface_Temperature", None),
                        getattr(keyparams, "Equilibrium_Temp", None)):
                if np.isfinite(_as_float(cand, np.nan)):
                    T_raw = cand
                    break

        # Depth fallback: use keyparams.Depth; if missing but you want *some* x-axis, you can
        # optionally fall back to the current probe index (commented out).
        D_raw = keyparams.Depth
        # if not np.isfinite(_as_float(D_raw, np.nan)):
        #     D_raw = keyparams.ProbeIndex  # <-- enable only if this makes scientific sense for you

        # Suitability fallback (you already had a proxy based on T)
        _suit = keyparams.Suitability
        if _suit is None or (isinstance(_suit, float) and not np.isfinite(_suit)):
            Ttmp = _as_float(T_raw, np.nan)
            _suit = 1.0 - min(1.0, abs(Ttmp - 273.15) / 200.0) if np.isfinite(Ttmp) else np.nan

        # ---- validation checks before append ----
        ctx = f"(probe={int(keyparams.ProbeIndex)}, iter={int(ii)})"
        T = check_range("Temperature", T_raw, *RANGES["Temperature"], context=ctx)
        P = check_range("Pressure", keyparams.Pressure, *RANGES["Pressure"], context=ctx)
        A = check_range("Bond_Albedo", keyparams.Bond_Albedo, *RANGES["Bond_Albedo"], context=ctx)
        GW = check_range("GreenhouseWarming", keyparams.GreenhouseWarming, *RANGES["GreenhouseWarming"], context=ctx)
        D = check_range("Depth", probe_coord, *RANGES["Depth"], context=ctx)
        S = check_range("Suitability", _suit, *RANGES["Suitability"], context=ctx)

        # Clean values to ensure no NaN/inf get through
        T_clean = safe_clean_value(T, fallback=273.15)  # Default to freezing point
        P_clean = safe_clean_value(P, fallback=1.0)     # Default to 1 atm
        A_clean = safe_clean_value(A, fallback=0.3)     # Default albedo
        GW_clean = safe_clean_value(GW, fallback=0.0)   # Default no greenhouse
        D_clean = safe_clean_value(D, fallback=probe_coord)  # Use probe coord as fallback
        S_clean = safe_clean_value(S, fallback=0.0)     # Default no suitability

        # Append cleaned values
        Temperature_Distribution.append(T_clean)
        Pressure_Distribution.append(P_clean)
        BondAlbedo_Distribution.append(A_clean)
        GreenHouse_Distribution.append(GW_clean)
        Depth_Distribution.append(D_clean)
        Suitability_Distribution.append(S_clean)
        runid = keyparams.runid

    print('Monte Carlo loop completed')
    print('Runid: ' + keyparams.runid)

    This_Suitability = np.mean(Suitability_Distribution)
    print('Average Suitability %.2f' % This_Suitability)

    Suitability_Plot.append(This_Suitability)
    Variable.append(probe_coord)
    SavedParameters.append(keyparams)

# ======================================
# RESULTS EXPORT FOR DASHBOARD
# ======================================

# Build a graph for layout calculation from the visual edges
DG_for_layout = nx.DiGraph()
DG_for_layout.add_nodes_from(range(len(Modules)))  # Include isolated nodes
DG_for_layout.add_edges_from(G.visual)


# Node names (fallback to mod_i)
nodes = [getattr(m, "name", f"mod_{i}") for i, m in enumerate(Modules)]


def _idx(i):
    """
    Convert numpy integer types to plain Python int.
    
    Args:
        i: Index value (may be numpy int64, float, etc.)
        
    Returns:
        int: Plain Python integer
    """
    try:
        return int(i) if isinstance(i, np.integer) else (int(i) if isinstance(i, (float,)) and i.is_integer() else i)
    except Exception:
        return i

def _name(i):
    """
    Get module name from index.
    
    Args:
        i: Module index
        
    Returns:
        str: Module name
    """
    j = _idx(i)
    try:
        return nodes[int(j)]
    except Exception:
        return str(j)

# Edges with labels, using human-readable names (no np.int64 in JSON)
edges = []
for (u, v) in DG_for_layout.edges():
    lbl = edge_labels.get((_idx(u), _idx(v)), "")
    edges.append((_name(u), _name(v), lbl))

# Quick spring layout for positions
pos = nx.spring_layout(DG_for_layout, seed=42)
positions = { _name(n): [float(pos[n][0]), float(pos[n][1])] for n in DG_for_layout.nodes() }

def _to_hex(c):
    """
    Convert color to hex string.
    
    Args:
        c: Color value (various formats)
        
    Returns:
        str: Hex color string
    """
    try:
        return _mcolors.to_hex(c, keep_alpha=False)
    except Exception:
        return str(c)

# Theme fallbacks (in case not set upstream)
prior_node_color       = globals().get("prior_node_color", "#1f77b4")      # blue
metabolism_node_color  = globals().get("metabolism_node_color", "#2ecc71") # green
other_node_color       = globals().get("other_node_color", "#e74c3c")      # red

# --- build node_colors to mirror the Matplotlib logic used in visualize() ---
node_colors = {}
for i, m in enumerate(Modules):
    name = getattr(m, "name", f"mod_{i}")
    # explicit override if present on the module
    explicit = getattr(m, "viz_color", None) or getattr(m, "color", None) or getattr(m, "Color", None)
    if explicit:
        node_colors[name] = _to_hex(explicit)
        continue

    in_params  = getattr(m, "input_parameters", [])  or []
    out_params = getattr(m, "output_parameters", []) or []
    if len(in_params) == 0:
        col = prior_node_color
    elif "Suitability" in out_params:
        col = metabolism_node_color
    else:
        col = other_node_color
    node_colors[name] = _to_hex(col)

def _resolve_logo_path(logo_rel: str):
    """
    Resolve planet logo image path from relative path.
    
    Searches multiple common locations for the logo file.
    
    Args:
        logo_rel: Relative path to logo from config
        
    Returns:
        str or None: Absolute path to logo file, or None if not found
    """
    if not logo_rel:
        return None
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, logo_rel),
        os.path.join(base, 'Habitats', 'Logos', logo_rel),
        os.path.join(base, 'Habitats', 'Logos', os.path.basename(logo_rel)),
        os.path.join(base, 'Figures', os.path.basename(logo_rel)),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

planet_image_path = _resolve_logo_path(str(HabitatLogo))

# helpful debug so you can see what was exported
print(f"[EXPORT] node_colors={len(node_colors)}  planet_image={'yes' if planet_image_path else 'no'}")

def _to_float_list(arr):
    """
    Convert array to list of floats, replacing invalid values with 0.0.
    
    Args:
        arr: Input array or list
        
    Returns:
        list: List of finite float values
    """
    out = []
    for x in arr:
        try:
            val = float(x)
            # Replace NaN/inf with reasonable defaults
            if not np.isfinite(val):
                val = 0.0
            out.append(val)
        except Exception:
            out.append(0.0)
    return out

probe_label = config.get('Probes', 'Label', fallback='Depth')
probe_unit  = config.get('Probes', 'Units', fallback='m')
temp_unit   = config.get('Outputs', 'TemperatureUnits', fallback='K')
suit_unit   = 'unitless'

node_colors = {}
try:
    # 1) Preferred: GraphVisualization attaches a dict of colors
    base_colors = getattr(G, "node_colors", None) or getattr(G, "NodeColors", None)
    if isinstance(base_colors, dict):
        for n in DG_for_layout.nodes():
            nm = _name(n)  # your helper maps index -> module name
            col = base_colors.get(n, base_colors.get(nm))
            if col:
                node_colors[nm] = str(col)

    # 2) Fallback: each module may carry a color attribute
    if not node_colors:
        for i, m in enumerate(Modules):
            nm = getattr(m, "name", f"mod_{i}")
            col = getattr(m, "color", None) or getattr(m, "Color", None)
            if col:
                node_colors[nm] = str(col)
except Exception:
    pass

# --- planet image (logo) path from the current config ---
planet_image_path = _resolve_logo_path(str(HabitatLogo))

results_payload = {
    "meta": {
        "config_used": str(config_file_path),
        "habitat_shortname": str(HabitatShortName),
        "n_modules": int(len(nodes)),
        "labels": {
            "temperature": {"label": "Temperature", "unit": temp_unit},
            "depth":       {"label": probe_label,  "unit": probe_unit},
            "suitability": {"label": "Suitability","unit": suit_unit},
        },
        "planet_image": str(planet_image_path) if planet_image_path else None,  
        # ---- diagnostics surfaced to the dashboard ----
        "diagnostics": {
            "n_warnings": int(len(diagnostics.get("warnings", []))) if 'diagnostics' in globals() else 0,
            "invalid_counts": diagnostics.get("invalid_counts", {}) if 'diagnostics' in globals() else {},
            "warnings_sample": (diagnostics.get("warnings", [])[:100] if 'diagnostics' in globals() else []),
        },
    },
    "graph": {
        "nodes": [str(n) for n in nodes],
        "edges": edges,
        "positions": positions,
        "directed": True,
        "node_colors": node_colors,
    },
    "distributions": {
        "temperature": _to_float_list(Temperature_Distribution),
        "depth": _to_float_list(Depth_Distribution),
        "suitability": _to_float_list(Suitability_Distribution),
    },
}



def _finite_count(lst):
    """
    Count the number of finite values in a list.
    
    Args:
        lst: Input list
        
    Returns:
        int: Count of finite values
    """
    a = np.array(lst, dtype=float)
    return int(np.isfinite(a).sum())

print("[EXPORT] Samples ->",
      "T:", len(Temperature_Distribution), f"finite={_finite_count(Temperature_Distribution)} |",
      "D:", len(Depth_Distribution),       f"finite={_finite_count(Depth_Distribution)} |",
      "S:", len(Suitability_Distribution), f"finite={_finite_count(Suitability_Distribution)}")
results_dir = os.path.join(os.path.dirname(__file__), "results")
print("[EXPORT] writing JSON to:", results_dir)  # optional debug
write_results_for_dashboard(results_payload, out_dir=results_dir)

write_results_for_dashboard(results_payload, out_dir="results")

# ======================================
# GRAPH VISUALIZATION
# ======================================

fig = plt.figure(figsize=(12.00, 8.00), dpi=300)
fig.set_facecolor(bkgcolor)
fig.set_edgecolor(selected_edgecolor)

ax = G.visualize()

# Add habitat logo
logo_path = _resolve_logo_path(str(HabitatLogo))
im = None
if logo_path and os.path.isfile(logo_path):
    try:
        im = plt.imread(logo_path)
    except Exception:
        im = None
if im is not None:
    newax = fig.add_axes([0.75, 0.75, 0.10, 0.10], anchor='NE')
    newax.set_axis_off()
    newax.imshow(im)

figures_dir = os.path.join(os.path.dirname(__file__), "Figures")
os.makedirs("Figures", exist_ok=True)

fig.savefig(os.path.join(figures_dir, HabitatShortName + '_Connections.png'))
fig.savefig(os.path.join(figures_dir, HabitatShortName + '_Connections.svg'))
plt.show()

# ======================================
# RESULTS VISUALIZATION
# ======================================

VisualizationModule(
    screen, sf, Suitability_Distribution, Temperature_Distribution,
    BondAlbedo_Distribution, GreenHouse_Distribution, Pressure_Distribution,
    Depth_Distribution, keyparams.runid, Suitability_Plot, Variable, HabitatLogo
)
