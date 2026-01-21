# batch_esmfold_and_graph.py
import time
import numpy as np
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
import logging
import sys
from pathlib import Path

# ---- Robust patch for ESMFold short-sequence TM/PTM crash (transformers 4.57.3 safe) ----
import transformers.models.esm.openfold_utils.loss as of_loss
import transformers.models.esm.modeling_esmfold as esmfold_mod

_ORIG_COMPUTE_TM = of_loss.compute_tm

def safe_compute_tm(ptm_logits, max_bin=31, no_bins=None):
    """
    Workaround for IndexError in compute_tm on very short sequences (e.g., 9-mers).
    If it still breaks, return NaN instead of crashing.
    """
    try:
        # For very short sequences, TM/PTM is not meaningful.
        # If L is too small, just return NaN.
        if hasattr(ptm_logits, "shape") and len(ptm_logits.shape) >= 3:
            L = int(ptm_logits.shape[-3])  # common layout uses ... x L x L x bins
            if L < 2:
                return ptm_logits.new_tensor(float("nan"))
        return _ORIG_COMPUTE_TM(ptm_logits, max_bin=max_bin, no_bins=no_bins)
    except IndexError:
        return ptm_logits.new_tensor(float("nan"))

# Patch both the original module and the copied reference inside modeling_esmfold
of_loss.compute_tm = safe_compute_tm
esmfold_mod.compute_tm = safe_compute_tm

print("[patch] compute_tm patched in loss:", of_loss.compute_tm is safe_compute_tm)
print("[patch] compute_tm patched in modeling_esmfold:", esmfold_mod.compute_tm is safe_compute_tm)



# -------------------------
# logging
# -------------------------
def setup_logging(
    log_file=None,
    level=logging.INFO,
    name=None,
):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 防止重复打印

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


logger = setup_logging()

logger.info(f"Patched compute_tm: {of_loss.compute_tm is safe_compute_tm}")

# -------------------------
# sequence utils
# -------------------------
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA20)}


def clean_seq(seq: str) -> str:
    seq = (seq or "").strip().upper()
    seq = "".join(seq.split())
    seq = "".join([c for c in seq if c in AA_TO_IDX])  # 只保留标准20AA
    return seq


def to_esm_tokens(seq: str) -> str:
    return " ".join(list(seq))


# -------------------------
# PDB -> graph utils
# -------------------------
def parse_pdb_ca_from_string(pdb_str: str):
    residues = []
    coords = []
    chains = []
    seen = set()
    for line in pdb_str.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        resname = line[17:20].strip()
        chain = line[21].strip() or "A"
        resseq = line[22:26].strip()
        icode = line[26].strip()
        key = (chain, resseq, icode)
        if key in seen:
            continue
        seen.add(key)
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        residues.append((resname, chain, resseq, icode))
        coords.append([x, y, z])
        chains.append(chain)
    return residues, np.asarray(coords, dtype=np.float32), chains


def resname_3to1(res3: str) -> str:
    m = {
        "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
        "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
        "THR":"T","VAL":"V","TRP":"W","TYR":"Y"
    }
    return m.get(res3.upper(), "X")


def build_graph(pdb_str: str, distance_cutoff: float = 8.0):
    residues, coords, chains = parse_pdb_ca_from_string(pdb_str)
    N = coords.shape[0]
    if N == 0:
        raise ValueError("No CA atoms parsed from PDB output")

    node_feat = np.zeros((N, 20), dtype=np.float32)
    res_id = []
    for i, (res3, chain, resseq, icode) in enumerate(residues):
        aa1 = resname_3to1(res3)
        if aa1 in AA_TO_IDX:
            node_feat[i, AA_TO_IDX[aa1]] = 1.0
        res_id.append(f"{chain}:{resseq}{icode}".strip())
    res_id = np.asarray(res_id)

    edges = []
    etypes = []

    # sequence edges (type 0)
    for i in range(N - 1):
        if chains[i] == chains[i + 1]:
            edges.append((i, i + 1)); etypes.append(0)
            edges.append((i + 1, i)); etypes.append(0)

    # spatial edges (type 1)
    cutoff2 = distance_cutoff * distance_cutoff
    for i in range(N):
        xi, yi, zi = coords[i]
        for j in range(i + 1, N):
            dx = xi - coords[j, 0]
            dy = yi - coords[j, 1]
            dz = zi - coords[j, 2]
            if dx*dx + dy*dy + dz*dz < cutoff2:
                edges.append((i, j)); etypes.append(1)
                edges.append((j, i)); etypes.append(1)

    edge_index = np.asarray(edges, dtype=np.int64).T  # (2, E)
    edge_type = np.asarray(etypes, dtype=np.int8)     # (E,)

    return node_feat, coords, edge_index, edge_type, res_id


# -------------------------
# ESMFold batch runner
# -------------------------
def load_esmfold(device: str, dtype: str):
    logger.info("Loading ESMFold tokenizer/model (once)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval()

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    model = model.to(device=device, dtype=torch_dtype)

    # ---- IMPORTANT: disable TM/PTM computation to avoid short-peptide crash ----
    disabled = False

    # Some versions keep it here
    if hasattr(model, "esmfold_config") and hasattr(model.esmfold_config, "compute_tm"):
        model.esmfold_config.compute_tm = False
        disabled = True

    # Some versions keep it here
    if hasattr(model, "config") and hasattr(model.config, "esmfold_config") and hasattr(model.config.esmfold_config, "compute_tm"):
        model.config.esmfold_config.compute_tm = False
        disabled = True

    # Fallback: some versions use compute_tm directly on config
    if hasattr(model, "config") and hasattr(model.config, "compute_tm"):
        model.config.compute_tm = False
        disabled = True

    logger.info(f"compute_tm disabled = {disabled}")

    return tokenizer, model


def _sync_if_cuda(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def fold_batch(
    seqs,
    tokenizer,
    model,
    device: str = "cuda",
    max_batch_size: int = 10,
):
    """
    Fold sequences in mini-batches (to avoid OOM).
    Returns list of dict: {seq_clean, pdb_str, mean_plddt, seconds}
    """
    results = []

    # clean & filter
    cleaned = []
    for s in seqs:
        c = clean_seq(s)
        if len(c) >= 3:
            cleaned.append(c)
        else:
            results.append({
                "seq_clean": c,
                "pdb_str": None,
                "mean_plddt": None,
                "seconds": 0.0,
                "error": f"too_short(len={len(c)})"
            })

    # process in chunks
    for start in range(0, len(cleaned), max_batch_size):
        chunk = cleaned[start:start + max_batch_size]
        chunk_tok = [to_esm_tokens(s) for s in chunk]

        inputs = tokenizer(chunk_tok, return_tensors="pt", add_special_tokens=False)

        ids = inputs.get("input_ids", None)
        if ids is None or ids.numel() == 0:
            raise ValueError("Tokenizer produced empty input_ids")

        # sanity check to avoid CUDA embedding assert
        vmin = int(ids.min().item())
        vmax = int(ids.max().item())
        if vmin < 0 or vmax >= tokenizer.vocab_size:
            raise ValueError(f"Bad token ids: min={vmin} max={vmax} vocab={tokenizer.vocab_size}")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        _sync_if_cuda(device)
        t0 = time.time()
        out = model(**inputs)
        _sync_if_cuda(device)
        dt = time.time() - t0

        # output_to_pdb returns list aligned with batch
        pdb_list = model.output_to_pdb(out)

        # pLDDT (shape usually [B, L] or similar)
        plddt = None
        for key in ["plddt", "predicted_lddt", "lddt"]:
            if hasattr(out, key):
                plddt = getattr(out, key)
                break
            if isinstance(out, dict) and key in out:
                plddt = out[key]
                break

        for i, seq_clean in enumerate(chunk):
            mean_plddt = None
            if plddt is not None:
                try:
                    # 常见是 (B, L)
                    if hasattr(plddt, "dim") and plddt.dim() >= 2:
                        mean_plddt = float(plddt[i].mean().item())
                    else:
                        mean_plddt = float(plddt.mean().item())
                except Exception:
                    mean_plddt = None

            results.append({
                "seq_clean": seq_clean,
                "pdb_str": pdb_list[i],
                "mean_plddt": mean_plddt,
                "seconds": dt,   # NOTE: chunk time; you也可以按 dt/len(chunk) 平均分摊
                "error": None
            })

        logger.info(f"Folded chunk: size={len(chunk)} time={dt:.2f}s")

    return results


def pldddt_i_mean(plddt_tensor, i: int) -> float:
    # Handles plddt_tensor shapes like (B, L) or dict-like
    t = plddt_tensor
    if isinstance(t, (list, tuple)):
        t = t[i]
    elif hasattr(t, "dim"):
        if t.dim() >= 2:
            t = t[i]
    return float(t.mean().item())


# -------------------------
# main benchmark
# -------------------------
def main():
    start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 如果你真在 GPU 上测吞吐，建议 float16；CPU 就 float32
    dtype = "float16" if device == "cuda" else "float32"

    logger.info(f"Device={device} dtype={dtype}")

    tokenizer, model = load_esmfold(device=device, dtype=dtype)

    # ---------
    # 你要测的 batch：10/100/...
    # 这里我用同一条序列复制来做纯性能基准（避免数据差异）
    # 你也可以换成真实序列列表
    base_seq = "GILGFVFTL"
    test_sizes = [10]  # 你可以加 [1, 10, 50, 100, 200] 等
    max_batch_size = 10     # 控制“每次送进模型”的batch，避免OOM

    # warmup（不计入正式benchmark）
    logger.info("Warmup...")
    _ = fold_batch([base_seq] * min(4, max_batch_size), tokenizer, model, device=device, max_batch_size=max_batch_size)

    # 是否保存 PDB / Graph
    save_outputs = True
    build_graphs = True

    out_dir = Path("batch_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in test_sizes:
        seqs = [base_seq] * n
        logger.info("=" * 60)
        logger.info(f"Benchmark: n_seqs={n} max_batch_size={max_batch_size}")

        _sync_if_cuda(device)
        t0 = time.time()
        results = fold_batch(seqs, tokenizer, model, device=device, max_batch_size=max_batch_size)
        _sync_if_cuda(device)
        fold_total = time.time() - t0

        # 简单统计
        ok = [r for r in results if r["pdb_str"] is not None and r["error"] is None]
        err = [r for r in results if r["error"] is not None]

        logger.info(f"Fold done: total_time={fold_total:.2f}s ok={len(ok)} err={len(err)}")
        if len(ok) > 0:
            logger.info(f"Throughput: {len(ok)/fold_total:.3f} seq/s  | avg {fold_total/len(ok):.3f} s/seq")

        # 保存 & 构图（可选）
        if save_outputs:
            for i, r in enumerate(results):
                if r["pdb_str"] is None:
                    continue
                pdb_path = out_dir / f"n{n}_i{i:04d}.pdb"
                with open(pdb_path, "w") as f:
                    f.write(r["pdb_str"])

                if build_graphs:
                    try:
                        node_feat, coords, edge_index, edge_type, res_id = build_graph(r["pdb_str"], distance_cutoff=8.0)
                        npz_path = out_dir / f"n{n}_i{i:04d}_graph.npz"
                        np.savez_compressed(
                            npz_path,
                            node_feat=node_feat,
                            coords=coords,
                            edge_index=edge_index,
                            edge_type=edge_type,
                            res_id=res_id
                        )
                    except Exception as e:
                        logger.warning(f"Graph build failed for i={i}: {e}")

            logger.info(f"Saved outputs under: {out_dir.resolve()}")

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"All benchmarks finished. Total wall time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
