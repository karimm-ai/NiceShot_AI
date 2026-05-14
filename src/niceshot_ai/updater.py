import os, sys, json, shutil
import zipfile
import tempfile
import urllib.request

REPO_ZIP_URL = "https://github.com/karimm-ai/NiceShot_AI/archive/refs/heads/main.zip"
VERSION_URL  = "https://raw.githubusercontent.com/karimm-ai/NiceShot_AI/main/version.json"

# Root of the tool folder (where niceshot_ai.py lives)
current_dir = os.path.dirname(os.path.abspath(__file__))
print(os.path.abspath(__file__))

while os.path.basename(current_dir) != "NiceShot_AI":
    current_dir = os.path.dirname(current_dir)

TOOL_DIR = current_dir


def get_current_version():
    """Read version from local version.json next to this script."""
    version_file = os.path.join(TOOL_DIR, "version.json")
    try:
        with open(version_file, "r") as f:
            return json.load(f)["version"]
    except Exception:
        return "0.0.0"  # force update if version file is missing


def get_latest_version():
    """Fetch latest version from GitHub."""
    try:
        with urllib.request.urlopen(VERSION_URL, timeout=5) as r:
            return json.loads(r.read().decode())["version"]
    except Exception:
        return None  # no internet or GitHub down — skip update silently


def parse_version(v):
    """Convert '1.2.3' to (1, 2, 3) for comparison."""
    try:
        return tuple(int(x) for x in v.strip().split("."))
    except Exception:
        return (0, 0, 0)


def download_and_apply_update():
    """Download latest ZIP from GitHub and replace current files."""
    print("[Updater] Downloading latest version...")

    tmp_zip = os.path.join(tempfile.gettempdir(), "NiceShot_AI_update.zip")
    tmp_dir = os.path.join(tempfile.gettempdir(), "NiceShot_AI_update")

    # Download ZIP
    try:
        urllib.request.urlretrieve(REPO_ZIP_URL, tmp_zip)
    except Exception as e:
        print(f"[Updater] FAILED: Could not download update: {e}")
        return False

    # Extract ZIP
    try:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        with zipfile.ZipFile(tmp_zip, "r") as z:
            z.extractall(tmp_dir)
    except Exception as e:
        print(f"[Updater] FAILED: Could not extract update: {e}")
        return False

    # Find the inner folder GitHub puts inside the ZIP (e.g., NiceShot_AI-main)
    extracted_folders = os.listdir(tmp_dir)
    if not extracted_folders:
        print("[Updater] FAILED: Update ZIP was empty.")
        return False

    # This will be the root of the extracted repo (GitHub puts this inside a folder)
    extracted_root_dir = os.path.join(tmp_dir, extracted_folders[0])

    # Copy all files from the extracted folder to the TOOL_DIR (root of your project)
    try:
        for item in os.listdir(extracted_root_dir):
            src = os.path.join(extracted_root_dir, item)
            dest = os.path.join(TOOL_DIR, item)
            
            # Handle directories
            if os.path.isdir(src):
                if os.path.exists(dest):
                    shutil.rmtree(dest)  # Remove the existing directory
                shutil.copytree(src, dest)  # Copy the entire directory
            else:
                # Handle individual files
                shutil.copy2(src, dest)  # Copy file with metadata
    except Exception as e:
        print(f"[Updater] FAILED: Could not apply update files: {e}")
        return False

    # Cleanup
    try:
        os.remove(tmp_zip)
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    print("[Updater] Update applied successfully!")
    return True


def check_and_update():
    """
    Main entry point. Call this at the top of niceshot_ai.py.
    Blocks until update is done, then restarts the process.
    """
    current = get_current_version()
    print(f"[Updater] Current version: v{current}")

    latest = get_latest_version()
    if latest is None:
        print("[Updater] Could not reach GitHub. Skipping update check.")
        return

    if parse_version(latest) <= parse_version(current):
        print(f"[Updater] Already up to date (v{current}).")
        return

    print(f"[Updater] New version available: v{latest}. Updating...")
    success = download_and_apply_update()

    if success:
        print(f"[Updater] Updated to v{latest}. Restarting...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    else:
        print("[Updater] Update failed. Running current version.")
    
    return success