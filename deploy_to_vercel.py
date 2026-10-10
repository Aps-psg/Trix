"""QueueFlux — Automated Vercel Deployment Script
================================================
Deploys the Canteen Digital Twin Web Interface directly to Vercel
using the Vercel REST API v13.
"""

import os
import sys
import json
import base64
from pathlib import Path
import urllib.request
import urllib.error


def deploy(token: str = None, project_name: str = "queueflux-control-tower"):
    if not token:
        token = os.environ.get("VERCEL_TOKEN")

    if not token and len(sys.argv) > 1:
        token = sys.argv[1].strip()

    if not token:
        print("=" * 70)
        print(" VERCEL DEPLOYMENT CONFIGURATION REQUIRED")
        print("=" * 70)
        print("Vercel requires authentication to link the deployment to your account.")
        print()
        print("Option 1 (Fastest - 1 Click in Browser):")
        print("  Click this pre-configured link to deploy immediately:")
        print("  -> https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FAps-psg%2FTrix")
        print()
        print("Option 2 (CLI / Automated Deployment via this script):")
        print("  1. Go to https://vercel.com/account/tokens")
        print("  2. Click 'Create Token', give it a name (e.g., 'QueueFlux')")
        print("  3. Run: python deploy_to_vercel.py <YOUR_TOKEN>")
        print("=" * 70)
        return False

    base_dir = Path(__file__).parent
    index_path = base_dir / "index.html"
    if not index_path.exists():
        index_path = base_dir / "frontend" / "index.html"

    if not index_path.exists():
        print(f"Error: index.html not found in {base_dir}")
        return False

    with open(index_path, "r", encoding="utf-8") as f:
        index_content = f.read()

    vercel_json_path = base_dir / "vercel.json"
    vercel_json_content = "{}"
    if vercel_json_path.exists():
        with open(vercel_json_path, "r", encoding="utf-8") as f:
            vercel_json_content = f.read()

    print(f"Preparing deployment payload for '{project_name}'...")
    
    # Vercel v13 API accepts files directly with utf-8 encoding
    files_payload = [
        {
            "file": "index.html",
            "data": index_content,
            "encoding": "utf-8"
        },
        {
            "file": "vercel.json",
            "data": vercel_json_content,
            "encoding": "utf-8"
        }
    ]

    payload = {
        "name": project_name,
        "files": files_payload,
        "projectSettings": {
            "framework": None
        }
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.vercel.com/v13/deployments",
        data=req_data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "QueueFlux-Deployer/2.0"
        },
        method="POST"
    )

    print("Uploading to Vercel API...")
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            url = data.get("url")
            print("=" * 70)
            print(" [SUCCESS] DEPLOYMENT SUCCESSFUL!")
            print("=" * 70)
            print(f"  Live Preview URL: https://{url}")
            if "alias" in data and data["alias"]:
                print(f"  Production Alias: https://{data['alias'][0]}")
            print(f"  Deployment ID:   {data.get('id')}")
            print(f"  Status:          {data.get('readyState')}")
            print("=" * 70)
            return True

    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        print(f"Deployment failed (HTTP {e.code}):")
        print(err_body)
        return False
    except Exception as e:
        print(f"Unexpected error during deployment: {e}")
        return False


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else None
    deploy(token=token_arg)
