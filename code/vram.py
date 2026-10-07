"""Optional GPU memory cap (HW2_VRAM_GB) and peak-memory report, used by the training scripts."""
import os


def apply_cap():
    gb = os.environ.get("HW2_VRAM_GB")
    if gb:
        import torch
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(gb) * 2**30 / total))
        print(f"[vram] capped at {gb} GiB", flush=True)


def report(tag):
    import torch
    print(f"[vram:{tag}] peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
          f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB", flush=True)
