"""
Upload the Oneiros project files to a Modal Volume.

Usage:
    pip install modal
    modal setup
    py scripts/upload_to_modal.py
"""
import subprocess
import sys

VOLUME_NAME = "oneiros-data"

# Folders to upload (local path -> remote path inside volume)
UPLOADS = [
    ("data/splits",     "/project/data/splits"),
    ("data/mutation_pairs.json", "/project/data/mutation_pairs.json"),
    ("engine",          "/project/engine"),
    ("baseline",        "/project/baseline"),
    ("config",          "/project/config"),
    ("scripts",         "/project/scripts"),
    ("harness",         "/project/harness"),
]


def run(cmd: str):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr.strip()}")
        return False
    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    return True


def main():
    print("=" * 60)
    print(f"Uploading Oneiros project to Modal Volume: {VOLUME_NAME}")
    print("=" * 60)

    # Create volume if it doesn't exist
    run(f"modal volume create {VOLUME_NAME}")

    for local_path, remote_path in UPLOADS:
        print(f"\n  Uploading {local_path} -> {remote_path}")
        success = run(f"modal volume put {VOLUME_NAME} {local_path} {remote_path}")
        if not success:
            print(f"  [FAILED] Could not upload {local_path}")

    print("\n" + "=" * 60)
    print("Upload complete! Verify with:")
    print(f"  modal volume ls {VOLUME_NAME} /project/")
    print(f"\nTo start training:")
    print(f"  modal run scripts/modal_train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
