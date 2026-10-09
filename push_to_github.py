"""
push_to_github.py - Automated GitHub Push Script for QueueFlux

Target Repository: https://github.com/Aps-psg/Trix
Username: sbsureka
"""

import os
import sys
import subprocess
import shutil

REPO_URL = "https://github.com/Aps-psg/Trix.git"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_git():
    """Finds git executable on the system."""
    # Check PATH
    git_path = shutil.which("git")
    if git_path:
        return git_path

    # Common Windows installation locations
    common_paths = [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Git\cmd\git.exe"),
        os.path.expanduser(r"~\AppData\Local\GitHubDesktop\bin\git.exe"),
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p
    return None


def run_cmd(cmd_list, cwd=PROJECT_DIR):
    print(f"Running: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, cwd=cwd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print(res.stderr)
    return res.returncode == 0


def main():
    git_bin = find_git()
    if not git_bin:
        print("=========================================================================")
        print("ERROR: Git executable was not found on your system PATH.")
        print("=========================================================================")
        print("\nPlease follow these simple steps to push to https://github.com/Aps-psg/Trix:")
        print("\n1. Install Git for Windows from: https://git-scm.com/download/win")
        print("   OR use GitHub Desktop from: https://desktop.github.com/")
        print("\n2. In VS Code, open Source Control tab (Ctrl+Shift+G) -> 'Publish to GitHub'")
        print("   Select repository: Aps-psg/Trix")
        print("\n3. Or run these commands in Git Bash / Terminal:")
        print(f"   git init")
        print(f"   git remote add origin {REPO_URL}")
        print(f"   git add .")
        print(f"   git commit -m 'Add QueueFlux Intelligence Layer'")
        print(f"   git branch -M main")
        print(f"   git push -u origin main")
        print("=========================================================================")
        return

    print(f"Found Git at: {git_bin}")
    os.chdir(PROJECT_DIR)

    # Initialize git if needed
    if not os.path.exists(os.path.join(PROJECT_DIR, ".git")):
        run_cmd([git_bin, "init"])

    # Configure remote
    run_cmd([git_bin, "remote", "remove", "origin"])
    run_cmd([git_bin, "remote", "add", "origin", REPO_URL])

    # Stage files
    run_cmd([git_bin, "add", "."])

    # Commit
    run_cmd([git_bin, "commit", "-m", "Add QueueFlux Intelligence Layer & Module Integration"])

    # Set branch to main
    run_cmd([git_bin, "branch", "-M", "main"])

    print("\nAttempting to push to https://github.com/Aps-psg/Trix...")
    success = run_cmd([git_bin, "push", "-u", "origin", "main"])

    if success:
        print("\nSUCCESS! Code successfully pushed to https://github.com/Aps-psg/Trix")
    else:
        print("\nPush requires GitHub authentication.")
        print("Please authenticate using GitHub Desktop or VS Code Source Control panel (Ctrl+Shift+G).")


if __name__ == "__main__":
    main()
