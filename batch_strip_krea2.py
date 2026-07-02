"""
Batch strip DIT/UNET layers from Krea2 LoRAs, keeping only text-conditioning
(txtfusion) layers. Skips any file that doesn't match the Krea2 key signature.

Includes a fidelity-risk heuristic: flags files where the kept (txtfusion)
tensors make up an unusually small fraction of the original file, since those
are the LoRAs most likely to lose meaningful visual fidelity after stripping.

Usage: python batch_strip_krea2.py "D:\\path\\to\\lora\\folder"
"""

import sys
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path
import os

OUTPUT_SUFFIX = "_stripped"
DRY_RUN = False

# If the kept tensors are below this % of the original file's tensor bytes,
# flag the result as higher-risk for fidelity loss. Tune based on your own
# A/B testing -- this is a heuristic, not a guarantee.
RISK_THRESHOLD_PCT = 10.0

STRIP_PREFIXES_RAW = ['blocks', 'first', 'last_linear', 'last.linear', 'tmlp_', 'tproj_1']
PREFIX = 'diffusion_model.'
STRIP_PREFIXES = [PREFIX + x for x in STRIP_PREFIXES_RAW]
KREA2_SIGNATURE = 'diffusion_model.txtfusion.'


def is_krea2_lora(filepath):
    try:
        with safe_open(filepath, framework="pt", device="cpu") as f:
            return any(k.startswith(KREA2_SIGNATURE) for k in f.keys())
    except Exception as e:
        print(f"  [ERROR] Could not read {filepath.name}: {e}")
        return False


def strip_layers(input_file, output_file, remove_prefixes):
    """
    Strips tensors matching remove_prefixes. Returns counts and byte sizes
    for both kept and removed tensors, so callers can compute a kept-byte
    ratio as a fidelity-risk proxy.
    """
    tensors = {}
    removed_count, kept_count = 0, 0
    removed_bytes, kept_bytes = 0, 0

    with safe_open(input_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in f.keys():
            tensor = f.get_tensor(key)
            tensor_bytes = tensor.element_size() * tensor.nelement()

            if any(key.startswith(p) for p in remove_prefixes):
                removed_count += 1
                removed_bytes += tensor_bytes
                continue

            tensors[key] = tensor
            kept_count += 1
            kept_bytes += tensor_bytes

    save_file(tensors, output_file, metadata=metadata)
    return {
        "removed_count": removed_count,
        "kept_count": kept_count,
        "removed_bytes": removed_bytes,
        "kept_bytes": kept_bytes,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_strip_krea2.py <folder_path>")
        sys.exit(1)

    lora_dir = Path(sys.argv[1].strip('"'))

    if not lora_dir.exists() or not lora_dir.is_dir():
        print(f"ERROR: Folder not found: {lora_dir}")
        sys.exit(1)

    files = sorted(lora_dir.glob("*.safetensors"))
    if not files:
        print(f"No .safetensors files found in {lora_dir}")
        return

    print(f"Found {len(files)} .safetensors files in {lora_dir}\n")

    results = []
    for f in files:
        if f.stem.endswith(OUTPUT_SUFFIX):
            continue

        print(f"Checking: {f.name}")
        if not is_krea2_lora(f):
            print("  -> SKIP (not Krea2 architecture / no txtfusion keys found)\n")
            results.append({"name": f.name, "status": "skipped"})
            continue

        out_path = f.parent / f"{f.stem}{OUTPUT_SUFFIX}.safetensors"
        if out_path.exists():
            print(f"  -> SKIP (output already exists: {out_path.name})\n")
            results.append({"name": f.name, "status": "already_done"})
            continue

        before_mb = os.path.getsize(f) / (1024 * 1024)

        if DRY_RUN:
            print(f"  -> [DRY RUN] Would strip and write {out_path.name}\n")
            results.append({"name": f.name, "status": "dry_run", "before_mb": before_mb})
            continue

        stats = strip_layers(f, out_path, STRIP_PREFIXES)
        after_mb = os.path.getsize(out_path) / (1024 * 1024)
        reduction = 100 * (1 - after_mb / before_mb)

        # Fidelity-risk heuristic: what % of original tensor bytes survived?
        total_tensor_bytes = stats["kept_bytes"] + stats["removed_bytes"]
        kept_byte_pct = (
            100 * stats["kept_bytes"] / total_tensor_bytes
            if total_tensor_bytes > 0 else 0
        )
        is_risky = kept_byte_pct < RISK_THRESHOLD_PCT

        print(f"  -> Stripped: removed {stats['removed_count']} tensors, kept {stats['kept_count']}")
        print(f"  -> {before_mb:.2f} MB -> {after_mb:.2f} MB ({reduction:.1f}% reduction)")
        print(f"  -> Kept-byte ratio: {kept_byte_pct:.2f}%"
              f"{'  [!] LOW -- higher risk of fidelity loss, A/B test before relying on this' if is_risky else ''}")
        print()

        results.append({
            "name": f.name,
            "status": "stripped",
            "before_mb": before_mb,
            "after_mb": after_mb,
            "kept_byte_pct": kept_byte_pct,
            "risky": is_risky,
        })

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        if r["status"] == "stripped":
            flag = "  [!] LOW-FIDELITY RISK" if r["risky"] else ""
            print(f"[OK]      {r['name']}: {r['before_mb']:.1f}MB -> {r['after_mb']:.1f}MB "
                  f"(kept {r['kept_byte_pct']:.1f}%){flag}")
        elif r["status"] == "dry_run":
            print(f"[DRY RUN] {r['name']}: {r['before_mb']:.1f}MB (would strip)")
        elif r["status"] == "already_done":
            print(f"[SKIP]    {r['name']}: output already exists")
        else:
            print(f"[SKIP]    {r['name']}: not Krea2 architecture")

    risky_files = [r["name"] for r in results if r.get("risky")]
    if risky_files:
        print()
        print(f"NOTE: {len(risky_files)} file(s) flagged as higher-risk for fidelity loss "
              f"(kept-byte ratio under {RISK_THRESHOLD_PCT}%):")
        for name in risky_files:
            print(f"  - {name}")
        print("Recommend A/B testing these specifically before relying on the stripped version.")


if __name__ == "__main__":
    main()
