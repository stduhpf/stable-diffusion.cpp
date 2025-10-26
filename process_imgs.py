#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# Configuration: Add directories or files to blacklist here
BLACKLIST = [
    ".git",          # Example: blacklist .git directories
    ".cache",  # Example: blacklist node_modules
    "build",
    "buildhip",
    "ggml",
    "thirdparty",
    "preview.png"
    # Add more directories or files as needed
]

def is_blacklisted(path):
    """Check if the path or any of its parents are blacklisted."""
    path = Path(path).resolve()
    for part in path.parts:
        if part in BLACKLIST:
            return True
    return False

def process_png_files(root_dir):
    """Recursively find and process all .png files in root_dir."""
    for root, dirs, files in os.walk(root_dir):
        # Skip blacklisted directories
        dirs[:] = [d for d in dirs if not is_blacklisted(os.path.join(root, d))]
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                file_path = os.path.join(root, file)
                if not is_blacklisted(file_path):
                    print(f"Processing: {file_path}")
                    try:
                        subprocess.run(
                            ["./buildstats/bin/sd",
                            "-m", "models/aios/cyberrealistic_v90.safetensors", 
                            # "--diffusion-model", "models/diffusionmodels/flux-mini-q4_k.gguf", 
                            # "--taesd", "models/tae/taesd.safetensors",
                             "-p", "\"a cat\"",
                             "--preview", "proj", 
                            # "--vae-on-cpu",
                            "--vae", "models/vae/sdxl.vae.safetensors",
                            # "--vae", "models/vae/ae.safetensors",
                             "-i", file_path],
                            check=True,
                            stdout=open("output.csv", "a"),
                            text=True
                        )
                    except subprocess.CalledProcessError as e:
                        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    process_png_files(os.getcwd())
    print("Done. Results are in output.csv")
