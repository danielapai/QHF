# Checks for the latest software version by fetching from GitHub
# Compares against current version and notifies if an update is available

import requests

CURRENT_VERSION = "1.0.0"  # Manually set your version here

VERSION_URL = "https://raw.githubusercontent.com/<rrathore02->/<your-repo>/main/latest_version.json"

def check_for_update():
    try:
        response = requests.get(VERSION_URL, timeout=5)
        if response.status_code == 200:
            latest_version = response.json().get("latest", "")
            if latest_version and latest_version != CURRENT_VERSION:
                print(f"[UPDATE AVAILABLE] A new version ({latest_version}) is available.")
                print("Download it from: https://github.com/<your-username>/<your-repo>/releases/latest")
            else:
                print("[Up to date] You are using the latest version.")
        else:
            print("[Warning] Could not check for updates (server error).")
    except Exception as e:
        print(f"[Warning] Update check failed: {e}")
