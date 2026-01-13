"""
QHF Tool Launcher

This script provides a command-line interface for launching the QHF tool.
It allows users to select configuration files, edit/create configs using a GUI,
and automatically runs QHF.py with the selected configuration.

Features:
- Version checking
- User login/session management
- Configuration file selection
- GUI-based configuration editor
- Automatic QHF execution

Author: QHF Development Team
"""

# ======================================
# IMPORTS
# ======================================

from modules.version_checker import check_for_update      # Checks if user has the latest version
from modules.user_login import get_user_info              # Manages user info and session
from modules.logout_user import logout_user               # Allows users to logout
import os
import sys

# ======================================
# HELPER FUNCTIONS
# ======================================

def run_qhf_with_config(config_path: str):
    """
    Execute QHF.py with the specified configuration file.
    
    Args:
        config_path: Path to the configuration file
    """
    script_path = os.path.join(os.path.dirname(__file__), "QHF.py")
    if not os.path.isfile(script_path):
        print(f"❌ QHF.py not found at: {script_path}")
        return
    if not os.path.isfile(config_path):
        print(f"❌ Config not found: {config_path}")
        return
    print(f"\n🚀 Running QHF with: {config_path}\n")
    os.system(f'python "{script_path}" "{config_path}"')

def show_config_list(config_dir):
    """
    Display a list of available configuration files.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        list: Sorted list of configuration filenames
    """
    configs = [f for f in os.listdir(config_dir) if f.endswith(".cfg")]
    configs.sort()
    if not configs:
        print("⚠️  No .cfg files found in Configs/. Use 'E' to create one in the GUI.")
    else:
        print("\n🗂 Available Config Files:")
        for idx, cfg in enumerate(configs, 1):
            print(f"  [{idx}] {cfg}")
    return configs

# ======================================
# MAIN FUNCTION
# ======================================

def main():
    """
    Main entry point for the QHF launcher.
    
    Handles user interaction, configuration selection, and QHF execution.
    """
    print("="*50)
    print("        QHF TOOL LAUNCHER")
    print("="*50)

    # Step 1: Check for latest version (non-fatal)
    try:
        check_for_update()
    except Exception as e:
        print(f"[Warning] Update check failed: {e}")

    # Step 2: Get user name and email (from cache or prompt)
    try:
        name, email = get_user_info()
    except Exception:
        name, email = ("anonymous", "anonymous")

    # Step 3: Menu
    repo_root = os.path.dirname(__file__)
    config_dir = os.path.join(repo_root, "Configs")
    last_handoff = os.path.join(repo_root, ".last_saved_cfg.txt")
    os.makedirs(config_dir, exist_ok=True)

    # Show configs and choices
    configs = show_config_list(config_dir)
    print("\nOptions:")
    print("  [number]  Run that config")
    print("  [E]       Edit/Create config using GUI (auto-run on save)")
    print("  [logout]  Sign out")

    choice = input("\nEnter number / 'E' / 'logout': ").strip().lower()

    if choice == "logout":
        logout_user()
        print("✅ You have been logged out.")
        return

    if choice == "e":
        # Launch GUI
        gui_path = os.path.join(repo_root, "qhf_config_gui.py")
        if not os.path.isfile(gui_path):
            print(f"❌ GUI not found: {gui_path}")
            print("   Save the GUI file as qhf_config_gui.py in the repo root.")
            return

        # Clear previous handoff (if any)
        if os.path.exists(last_handoff):
            try:
                os.remove(last_handoff)
            except Exception:
                pass

        # Open the GUI (blocking)
        exit_code = os.system(f'python "{gui_path}"')
        if exit_code != 0:
            print("❌ GUI exited with an error or was closed unexpectedly.")
            return

        # After GUI closes, see if a config was saved
        if os.path.isfile(last_handoff):
            try:
                with open(last_handoff, "r") as f:
                    cfg_path = f.read().strip()
            except Exception:
                cfg_path = ""

            if cfg_path:
                print(f"🔗 Detected saved config from GUI: {cfg_path}")
                run_qhf_with_config(cfg_path)
                return
            else:
                print("ℹ️ GUI closed without saving a config. Nothing to run.")
                return
        else:
            print("ℹ️ No handoff file found. If you saved, ensure GUI wrote .last_saved_cfg.txt.")
            return

    # Else: number path
    try:
        idx = int(choice)
        configs = show_config_list(config_dir)
        if not configs:
            print("❌ No configs to run. Use 'E' to create one.")
            return
        config_name = configs[idx - 1]
    except (ValueError, IndexError):
        print("❌ Invalid selection.")
        return

    config_path = os.path.join(config_dir, config_name)

    # Preview a few lines (nice touch for sanity)
    try:
        with open(config_path, "r") as f:
            preview_lines = []
            for _ in range(5):
                try:
                    preview_lines.append(next(f))
                except StopIteration:
                    break
        print(f"\n[DEBUG] First {len(preview_lines)} lines of {config_path}:\n" + "".join(preview_lines))
    except Exception as e:
        print(f"[DEBUG] Could not preview config: {e}")

    # Run QHF
    run_qhf_with_config(config_path)

if __name__ == "__main__":
    main()