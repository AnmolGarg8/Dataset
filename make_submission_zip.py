"""
Builds the official Amazon ML Challenge 2026 submission zip package.
Ensures exact directory structure, no pycache or temporary files, and verifies contents.
"""

import os
import zipfile
import sys

def create_submission_zip(team_name: str = "EntityResolvers", output_dir: str = "."):
    zip_filename = os.path.join(output_dir, f"{team_name}_submission.zip")
    print(f"Creating submission package: {zip_filename}")

    # Files to include
    include_paths = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        ("Documentation_template.md", "Documentation_template.md"),
        ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
        ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
    ]

    src_dir = "code/business_entity_resolution/src"
    for fname in os.listdir(src_dir):
        if fname.endswith(".py"):
            full_path = os.path.join(src_dir, fname)
            arc_name = f"code/business_entity_resolution/src/{fname}"
            include_paths.append((full_path, arc_name))

    # Also include trained model artifact in code/business_entity_resolution/output/model.joblib if desired
    model_path = "code/business_entity_resolution/output/model.joblib"
    if os.path.exists(model_path):
        include_paths.append((model_path, "code/business_entity_resolution/output/model.joblib"))

    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for full_path, arcname in include_paths:
            if not os.path.exists(full_path):
                print(f"ERROR: Missing required file: {full_path}")
                sys.exit(1)
            file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  Adding {arcname} ({file_size_mb:.2f} MB)...")
            zf.write(full_path, arcname)

    total_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"\nSUCCESS: Created {zip_filename} ({total_size_mb:.2f} MB)")

if __name__ == "__main__":
    team = sys.argv[1] if len(sys.argv) > 1 else "EntityResolvers"
    create_submission_zip(team)
