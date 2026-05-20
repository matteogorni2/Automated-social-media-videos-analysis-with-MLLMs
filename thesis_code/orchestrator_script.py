#!/usr/bin/env python3
"""
submit_from_manifest.py

Generic HPC job submission wrapper.

- Reads a YAML manifest with tunable parameters.
- Reads an SBATCH template file with placeholders like ${PLACEHOLDER}.
- Substitutes manifest values into the template.
- Writes a temporary SBATCH file and submits it via sbatch.

This version makes a unique temporary copy of the manifest for each submission.
"""
import argparse
import yaml
import os
import sys
import tempfile
import subprocess
from string import Template
import shutil
import uuid

def main():
    parser = argparse.ArgumentParser(description="Submit HPC job from manifest and template")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to YAML manifest with job parameters"
    )
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        help="Path to SBATCH template file with placeholders"
    )
    args = parser.parse_args()

    # --- Load manifest (as dict) ---
    try:
        with open(args.manifest, 'r') as f:
            manifest = yaml.safe_load(f) or {}
            if not isinstance(manifest, dict):
                # If the YAML root is not a mapping, wrap it to avoid later errors.
                manifest = {"manifest_content": manifest}
    except Exception as e:
        print(f"Error reading manifest: {e}")
        sys.exit(1)

    # --- Create a per-submission temp directory and copy the manifest there ---
    # Base temp folder under user control
    BASE_TMP_DIR = "tmp_sbatch"
    os.makedirs(BASE_TMP_DIR, exist_ok=True)
    # Unique subfolder for this submission
    tmp_dir = os.path.join(BASE_TMP_DIR, f"sbatch_{uuid.uuid4().hex}")
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        manifest_basename = os.path.basename(args.manifest)
        #unique_name = f"{os.path.splitext(manifest_basename)[0]}_{uuid.uuid4().hex}.yaml"
        unique_name = "experiment_manifest.yaml"
        tmp_manifest_path = os.path.join(tmp_dir, unique_name)
        shutil.copy2(args.manifest, tmp_manifest_path)
    except Exception as e:
        print(f"Error creating temporary manifest copy: {e}")
        sys.exit(1)

    # --- Inject absolute path for the copied manifest (used inside the SBATCH template) ---
    manifest["manifest_absolute_path"] = os.path.abspath(tmp_manifest_path)

    # --- Read SBATCH template ---
    try:
        with open(args.template, 'r') as f:
            template_content = f.read()
    except Exception as e:
        print(f"Error reading template: {e}")
        sys.exit(1)

    # --- Substitute placeholders ---
    try:
        templ = Template(template_content)
        # Template expects mapping of strings; stringify top-level values to be safe.
        mapping = {k: str(v) for k, v in manifest.items()}
        sbatch_content = templ.safe_substitute(mapping)
    except KeyError as e:
        print(f"Missing key in manifest for template placeholder: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during template substitution: {e}")
        sys.exit(1)

    # --- Write temporary SBATCH file (in same tmp_dir so manifest and sbatch stay together) ---
    tmp_file = os.path.join(tmp_dir, "job.sbatch")
    try:
        with open(tmp_file, 'w') as f:
            f.write(sbatch_content)
    except Exception as e:
        print(f"Error writing sbatch file: {e}")
        sys.exit(1)

    print(f"Generated SBATCH file: {tmp_file}")
    print(f"Temporary manifest copy: {tmp_manifest_path}")
    print(f"Temporary directory retained so scheduler can access manifest until job runs: {tmp_dir}")

    # --- Submit job ---
    try:
        result = subprocess.run(
            ["sbatch", tmp_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if result.returncode == 0:
            print(f"Job submitted successfully: {result.stdout.strip()}")
        else:
            print(f"Error submitting job:\n{result.stderr}")
    except Exception as e:
        print(f"Error calling sbatch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

