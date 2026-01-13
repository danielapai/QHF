"""
QHF Config Editor (v2.1)
- Handles both single-file modules and packages (directories with __init__.py)
- Introspects public classes and functions (even if imported)
- Validates logo & numeric fields, supports layout shortname dropdown
- Loads last config on startup, Refresh button, Run console
"""
import os, sys, threading, queue, subprocess, configparser, importlib.util, inspect
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(REPO_ROOT, "Configs")
HABITATS_DIR = os.path.join(REPO_ROOT, "Habitats")
METAB_DIR    = os.path.join(REPO_ROOT, "Metabolisms")
ANALYSES_DIR = os.path.join(REPO_ROOT, "Analyses")
QHF_SCRIPT = os.path.join(REPO_ROOT, "QHF.py")
LAST_PATH_FILE = os.path.join(REPO_ROOT, ".last_saved_cfg.txt")

def list_py_modules(folder):
    items = []
    if not os.path.isdir(folder): return items
    for f in os.listdir(folder):
        full = os.path.join(folder, f)
        if f == "__pycache__": continue
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "__init__.py")):
            items.append(f)                      # package
        elif f.endswith(".py") and f != "__init__.py":
            items.append(os.path.splitext(f)[0]) # plain module
    items.sort()
    return items

def load_module_from_path(module_path):
    try:
        if not os.path.isfile(module_path):
            return None, "File not found"
        name = f"_tmp_{os.path.basename(module_path).replace('.', '_')}"
        spec = importlib.util.spec_from_file_location(name, module_path)
        if spec is None: return None, "Spec load failed"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod, None
    except Exception as e:
        return None, str(e)

def discover_symbols(module_base_path):
    """
    Return (classes, callables, err) for a .py file or a package dir.
    Public = not starting with '_' (we include imported symbols).
    """
    path = module_base_path
    if os.path.isdir(path):
        init_py = os.path.join(path, "__init__.py")
        if not os.path.isfile(init_py):
            return [], [], "Package missing __init__.py"
        load_path = init_py
    else:
        if not os.path.splitext(path)[1]:
            path = path + ".py"
        load_path = path

    mod, err = load_module_from_path(load_path)
    if mod is None:
        return [], [], err

    classes, funcs = [], []
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"):
            continue
        try:
            if inspect.isclass(obj): classes.append(name)
            elif inspect.isfunction(obj) or inspect.isbuiltin(obj): funcs.append(name)
        except Exception:
            continue
    # dedupe & sort
    return sorted(set(classes)), sorted(set(funcs)), None

def get_layout_shortnames():
    lp_path = os.path.join(REPO_ROOT, "layout_presets.py")
    mod, err = load_module_from_path(lp_path)
    if mod is None: return [], f"layout_presets import error: {err}"
    for key in ["PRESETS", "LAYOUT_PRESETS", "presets"]:
        if hasattr(mod, key) and isinstance(getattr(mod, key), dict):
            return sorted(list(getattr(mod, key).keys())), None
    return [], None

def read_cfg(path):
    cp = configparser.ConfigParser(); cp.read(path); return cp

def write_cfg(path, data):
    cp = configparser.ConfigParser()
    cp["Configuration"] = {"ConfigID": data["ConfigID"]}
    cp["Habitat"] = {
        "HabitatModule": data["HabitatModule"],
        "HabitatClass": data["HabitatClass"],
        "HabitatLogo": data["HabitatLogo"],
        "HabitatShortname": data["HabitatShortname"],
    }
    cp["Metabolism"] = {
        "MetabolismModule": data["MetabolismModule"],
        "MetabolismClass": data["MetabolismClass"],
    }
    cp["Visualization"] = {
        "VisualizationModule": data["VisualizationModule"],
        "VisualizationCallable": data["VisualizationCallable"],
    }
    cp["Sampling"] = {
        "NumProbes": str(data["NumProbes"]),
        "Niterations": str(data["Niterations"]),
    }
    with open(path, "w", encoding="utf-8") as f: cp.write(f)

class OutputConsole(tk.Toplevel):
    def __init__(self, master, title="QHF Output"):
        super().__init__(master); self.title(title); self.geometry("900x400")
        self.text = tk.Text(self, wrap="word", state="disabled"); self.text.pack(fill="both", expand=True)
        self._queue = queue.Queue(); self._after_id = None
        self.protocol("WM_DELETE_WINDOW", self.destroy)
    def append(self, s: str):
        self._queue.put(s)
        if self._after_id is None: self._after_id = self.after(50, self._drain)
    def _drain(self):
        try:
            while True:
                s = self._queue.get_nowait()
                self.text.configure(state="normal"); self.text.insert("end", s); self.text.see("end"); self.text.configure(state="disabled")
        except queue.Empty: pass
        self._after_id = None

class QHFConfigGUI(tk.Tk):
    def __init__(self):
        super().__init__(); self.title("QHF Config Editor v2.1"); self.geometry("1000x720")
        self.current_cfg_path = None
        self.v_configid = tk.StringVar(value="New configuration")
        self.v_hab_module = tk.StringVar(); self.v_hab_class = tk.StringVar(); self.v_hab_logo = tk.StringVar(); self.v_hab_short = tk.StringVar()
        self.v_metab_module = tk.StringVar(); self.v_metab_class = tk.StringVar()
        self.v_vis_module = tk.StringVar(); self.v_vis_callable = tk.StringVar()
        self.v_num_probes = tk.StringVar(value="100"); self.v_niterations = tk.StringVar(value="50")
        self._build_ui(); self.refresh_lists()
        if os.path.isfile(LAST_PATH_FILE):
            try:
                last = open(LAST_PATH_FILE, "r", encoding="utf-8").read().strip()
                if last and os.path.isfile(last): self.load_cfg(last)
            except Exception: pass

    def _build_ui(self):
        frm = ttk.Frame(self); frm.pack(fill="both", expand=True, padx=12, pady=12)
        hdr = ttk.Frame(frm); hdr.pack(fill="x", pady=(0,8))
        ttk.Label(hdr, text="Config ID").pack(side="left")
        ttk.Entry(hdr, textvariable=self.v_configid, width=40).pack(side="left", padx=8)
        ttk.Button(hdr, text="Load", command=self.on_load).pack(side="left", padx=4)
        ttk.Button(hdr, text="Save", command=self.on_save).pack(side="left", padx=4)
        ttk.Button(hdr, text="Save As", command=self.on_save_as).pack(side="left", padx=4)
        ttk.Button(hdr, text="Refresh", command=self.refresh_lists).pack(side="left", padx=4)
        ttk.Button(hdr, text="Validate", command=self.on_validate).pack(side="left", padx=12)
        ttk.Button(hdr, text="Run", command=self.on_run).pack(side="left", padx=4)
        nb = ttk.Notebook(frm); nb.pack(fill="both", expand=True)

        tab_h = ttk.Frame(nb); nb.add(tab_h, text="Habitat")
        row = ttk.Frame(tab_h); row.pack(fill="x", pady=6)
        ttk.Label(row, text="Habitat Module").pack(side="left", padx=(0,6))
        self.cb_hab_mod = ttk.Combobox(row, textvariable=self.v_hab_module, state="readonly", width=40); self.cb_hab_mod.pack(side="left")
        self.cb_hab_mod.bind("<<ComboboxSelected>>", self._on_select_hab_module)
        row = ttk.Frame(tab_h); row.pack(fill="x", pady=6)
        ttk.Label(row, text="Habitat Class").pack(side="left", padx=(0,6))
        self.cb_hab_class = ttk.Combobox(row, textvariable=self.v_hab_class, state="readonly", width=40); self.cb_hab_class.pack(side="left")
        row = ttk.Frame(tab_h); row.pack(fill="x", pady=6)
        ttk.Label(row, text="Habitat Logo").pack(side="left", padx=(0,6))
        ttk.Entry(row, textvariable=self.v_hab_logo, width=50).pack(side="left"); ttk.Button(row, text="Browse", command=self._pick_logo).pack(side="left", padx=6)
        row = ttk.Frame(tab_h); row.pack(fill="x", pady=6)
        ttk.Label(row, text="Habitat Shortname").pack(side="left", padx=(0,6))
        self.cb_hab_short = ttk.Combobox(row, textvariable=self.v_hab_short, width=20); self.cb_hab_short.pack(side="left")
        shorts, _ = get_layout_shortnames(); 
        if shorts: self.cb_hab_short["values"] = shorts

        tab_m = ttk.Frame(nb); nb.add(tab_m, text="Metabolism")
        row = ttk.Frame(tab_m); row.pack(fill="x", pady=6)
        ttk.Label(row, text="Metabolism Module").pack(side="left", padx=(0,6))
        self.cb_metab_mod = ttk.Combobox(row, textvariable=self.v_metab_module, state="readonly", width=40); self.cb_metab_mod.pack(side="left")
        self.cb_metab_mod.bind("<<ComboboxSelected>>", self._on_select_metab_module)
        row = ttk.Frame(tab_m); row.pack(fill="x", pady=6)
        ttk.Label(row, text="Metabolism Class").pack(side="left", padx=(0,6))
        self.cb_metab_class = ttk.Combobox(row, textvariable=self.v_metab_class, state="readonly", width=40); self.cb_metab_class.pack(side="left")

        tab_v = ttk.Frame(nb); nb.add(tab_v, text="Visualization")
        row = ttk.Frame(tab_v); row.pack(fill="x", pady=6)
        ttk.Label(row, text="Visualization Module").pack(side="left", padx=(0,6))
        self.cb_vis_mod = ttk.Combobox(row, textvariable=self.v_vis_module, state="readonly", width=40); self.cb_vis_mod.pack(side="left")
        self.cb_vis_mod.bind("<<ComboboxSelected>>", self._on_select_vis_module)
        row = ttk.Frame(tab_v); row.pack(fill="x", pady=6)
        ttk.Label(row, text="Visualization Callable/Class").pack(side="left", padx=(0,6))
        self.cb_vis_callable = ttk.Combobox(row, textvariable=self.v_vis_callable, state="readonly", width=40); self.cb_vis_callable.pack(side="left")

        tab_s = ttk.Frame(nb); nb.add(tab_s, text="Sampling")
        row = ttk.Frame(tab_s); row.pack(fill="x", pady=6)
        ttk.Label(row, text="NumProbes").pack(side="left", padx=(0,6))
        ttk.Entry(row, textvariable=self.v_num_probes, width=10).pack(side="left"); ttk.Label(row, text="(positive integer)").pack(side="left", padx=6)
        row = ttk.Frame(tab_s); row.pack(fill="x", pady=6)
        ttk.Label(row, text="Niterations").pack(side="left", padx=(0,6))
        ttk.Entry(row, textvariable=self.v_niterations, width=10).pack(side="left"); ttk.Label(row, text="(positive integer)").pack(side="left", padx=6)

        self.status = tk.StringVar(value="Ready."); ttk.Label(frm, textvariable=self.status, anchor="w").pack(fill="x", pady=(8,0))

    def refresh_lists(self):
        self.cb_hab_mod["values"] = list_py_modules(HABITATS_DIR)
        self.cb_metab_mod["values"] = list_py_modules(METAB_DIR)
        self.cb_vis_mod["values"] = list_py_modules(ANALYSES_DIR)
        self.status.set("Module lists refreshed.")
        if self.v_hab_module.get(): self._on_select_hab_module()
        if self.v_metab_module.get(): self._on_select_metab_module()
        if self.v_vis_module.get(): self._on_select_vis_module()

    def _on_select_hab_module(self, *_):
        modname = self.v_hab_module.get(); 
        if not modname: return
        base = os.path.join(HABITATS_DIR, modname)
        classes, _, err = discover_symbols(base)
        if err: messagebox.showwarning("Import warning", f"Could not inspect {modname}: {err}")
        self.cb_hab_class["values"] = classes or []
        if classes and self.v_hab_class.get() not in classes: self.v_hab_class.set(classes[0])

    def _on_select_metab_module(self, *_):
        modname = self.v_metab_module.get(); 
        if not modname: return
        base = os.path.join(METAB_DIR, modname)
        classes, _, err = discover_symbols(base)
        if err: messagebox.showwarning("Import warning", f"Could not inspect {modname}: {err}")
        self.cb_metab_class["values"] = classes or []
        if classes and self.v_metab_class.get() not in classes: self.v_metab_class.set(classes[0])

    def _on_select_vis_module(self, *_):
        modname = self.v_vis_module.get(); 
        if not modname: return
        base = os.path.join(ANALYSES_DIR, modname)
        classes, funcs, err = discover_symbols(base)
        if err: messagebox.showwarning("Import warning", f"Could not inspect {modname}: {err}")
        vals = (classes or []) + (funcs or [])
        self.cb_vis_callable["values"] = vals
        if vals and self.v_vis_callable.get() not in vals: self.v_vis_callable.set(vals[0])

    def _pick_logo(self):
        fn = filedialog.askopenfilename(title="Choose Habitat Logo", initialdir=REPO_ROOT,
                                        filetypes=[("Images", "*.png *.jpg *.jpeg *.svg *.gif"), ("All files", "*.*")])
        if fn: self.v_hab_logo.set(os.path.relpath(fn, REPO_ROOT))

    def _gather_data(self):
        def as_pos_int(s, default):
            try:
                v = int(str(s).strip()); 
                return v if v > 0 else default
            except Exception: return default
        return {
            "ConfigID": self.v_configid.get().strip() or "New configuration",
            "HabitatModule": self.v_hab_module.get().strip(),
            "HabitatClass": self.v_hab_class.get().strip(),
            "HabitatLogo": self.v_hab_logo.get().strip(),
            "HabitatShortname": self.v_hab_short.get().strip(),
            "MetabolismModule": self.v_metab_module.get().strip(),
            "MetabolismClass": self.v_metab_class.get().strip(),
            "VisualizationModule": self.v_vis_module.get().strip(),
            "VisualizationCallable": self.v_vis_callable.get().strip(),
            "NumProbes": as_pos_int(self.v_num_probes.get(), 100),
            "Niterations": as_pos_int(self.v_niterations.get(), 50),
        }

    def on_load(self):
        fn = filedialog.askopenfilename(title="Open .cfg", initialdir=CONFIGS_DIR, filetypes=[("QHF config", "*.cfg"), ("All files", "*.*")])
        if fn: self.load_cfg(fn)

    def load_cfg(self, path):
        cp = read_cfg(path)
        self.v_configid.set(cp.get("Configuration", "ConfigID", fallback="New configuration"))
        self.v_hab_module.set(cp.get("Habitat", "HabitatModule", fallback="")); self._on_select_hab_module()
        self.v_hab_class.set(cp.get("Habitat", "HabitatClass", fallback=""))
        self.v_hab_logo.set(cp.get("Habitat", "HabitatLogo", fallback=""))
        self.v_hab_short.set(cp.get("Habitat", "HabitatShortname", fallback=""))
        self.v_metab_module.set(cp.get("Metabolism", "MetabolismModule", fallback="")); self._on_select_metab_module()
        self.v_metab_class.set(cp.get("Metabolism", "MetabolismClass", fallback=""))
        self.v_vis_module.set(cp.get("Visualization", "VisualizationModule", fallback="")); self._on_select_vis_module()
        self.v_vis_callable.set(cp.get("Visualization", "VisualizationCallable", fallback=""))
        self.v_num_probes.set(cp.get("Sampling", "NumProbes", fallback="100"))
        self.v_niterations.set(cp.get("Sampling", "Niterations", fallback="50"))
        self.current_cfg_path = path
        with open(LAST_PATH_FILE, "w", encoding="utf-8") as f: f.write(path)
        self.status.set(f"Loaded: {os.path.relpath(path, REPO_ROOT)}")

    def on_save(self):
        if not self.current_cfg_path: return self.on_save_as()
        data = self._gather_data()
        ok, msg = self._validate(data, deep=False)
        if not ok and not messagebox.askyesno("Validation", f"{msg}\n\nSave anyway?"): return
        write_cfg(self.current_cfg_path, data)
        with open(LAST_PATH_FILE, "w", encoding="utf-8") as f: f.write(self.current_cfg_path)
        self.status.set(f"Saved: {os.path.relpath(self.current_cfg_path, REPO_ROOT)}")

    def on_save_as(self):
        fn = filedialog.asksaveasfilename(title="Save .cfg As", defaultextension=".cfg", initialdir=CONFIGS_DIR,
                                          filetypes=[("QHF config", "*.cfg"), ("All files", "*.*")])
        if not fn: return
        self.current_cfg_path = fn
        self.on_save()

    def _validate(self, data, deep=True):
        required = ["HabitatModule","HabitatClass","MetabolismModule","MetabolismClass","VisualizationModule","VisualizationCallable"]
        miss = [k for k in required if not data[k]]
        if miss: return False, f"Missing required fields: {', '.join(miss)}"

        def resolve(folder, base):
            p = os.path.join(folder, base)
            if os.path.isdir(p): p = os.path.join(p, "__init__.py")
            elif not os.path.splitext(p)[1]: p = p + ".py"
            return p

        for folder, base in [(HABITATS_DIR,data["HabitatModule"]),(METAB_DIR,data["MetabolismModule"]),(ANALYSES_DIR,data["VisualizationModule"])]:
            if not os.path.isfile(resolve(folder, base)):
                return False, f"Referenced module not found: {resolve(folder, base)}"

        if data["HabitatLogo"]:
            logo_abs = os.path.join(REPO_ROOT, data["HabitatLogo"])
            if not os.path.isfile(logo_abs): return False, f"HabitatLogo file not found: {logo_abs}"

        if data["NumProbes"] <= 0 or data["Niterations"] <= 0: return False, "NumProbes and Niterations must be positive integers."
        if not deep: return True, "Basic validation passed."

        # Deep symbol checks
        for (folder, base, name) in [
            (HABITATS_DIR, data["HabitatModule"], data["HabitatClass"]),
            (METAB_DIR, data["MetabolismModule"], data["MetabolismClass"]),
            (ANALYSES_DIR, data["VisualizationModule"], data["VisualizationCallable"]),
        ]:
            base_path = os.path.join(folder, base)
            classes, funcs, err = discover_symbols(base_path)
            if err: return False, f"Import error for {base}: {err}"
            if name not in classes + funcs: return False, f"Symbol '{name}' not found in '{base}'"

        return True, "Deep validation passed."

    def on_validate(self):
        data = self._gather_data()
        ok, msg = self._validate(data, deep=True)
        (messagebox.showinfo if ok else messagebox.showerror)("Validation", msg)

    def on_run(self):
        if not os.path.isfile(QHF_SCRIPT):
            messagebox.showerror("Run Error", f"QHF.py not found at:\n{QHF_SCRIPT}"); return
        if not self.current_cfg_path:
            if not messagebox.askyesno("Not saved", "Config not saved yet. Save now?"): return
            self.on_save_as()
            if not self.current_cfg_path: return
        data = self._gather_data()
        ok, msg = self._validate(data, deep=False)
        if not ok: messagebox.showerror("Validation", msg); return

        console = OutputConsole(self, title="QHF Output")
        try:
            cmd = [sys.executable, QHF_SCRIPT, self.current_cfg_path]
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            def reader(stream, prefix=""):
                for line in iter(stream.readline, ""):
                    console.append(prefix + line)
                stream.close()
            threading.Thread(target=reader, args=(proc.stdout, ""), daemon=True).start()
            threading.Thread(target=reader, args=(proc.stderr, "[stderr] "), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Run Error", str(e))

if __name__ == "__main__":
    app = QHFConfigGUI(); app.mainloop()
