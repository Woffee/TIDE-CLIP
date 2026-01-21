# single_esmfold_and_graph.py
import time
import numpy as np
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
import logging
import sys
from pathlib import Path

def setup_logging(
    log_file=None,
    level=logging.INFO,
    name=None,
):
    """
    Setup logging with timestamp.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 防止重复打印

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---- console handler ----
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # ---- file handler (optional) ----
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

logger = setup_logging()

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA20)}

def clean_seq(seq: str) -> str:
    seq = (seq or "").strip().upper()
    seq = "".join(seq.split())
    # 保守：只保留标准20AA
    seq = "".join([c for c in seq if c in AA_TO_IDX])
    return seq

def to_esm_tokens(seq: str) -> str:
    # ESM tokenizer 更稳的输入形式：氨基酸字符之间加空格
    return " ".join(list(seq))

@torch.inference_mode()
def fold_one(seq: str, device: str = "cuda", dtype: str = "float16"):
    seq = clean_seq(seq)
    if len(seq) < 3:
        raise ValueError(f"Sequence too short after cleaning: len={len(seq)} seq='{seq}'")

    # load model/tokenizer
    logger.info("Loading ESMFold model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval()

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    model = model.to(device=device, dtype=torch_dtype)

    # tokenize (IMPORTANT)
    logger.info("Tokenizing input sequence...")
    seq_tok = to_esm_tokens(seq)
    inputs = tokenizer([seq_tok], return_tensors="pt", add_special_tokens=False)

    ids = inputs.get("input_ids", None)
    if ids is None or ids.numel() == 0:
        raise ValueError("Tokenizer produced empty input_ids")
    # token id sanity check (avoid CUDA device-side assert in embeddings)
    vmin = int(ids.min().item())
    vmax = int(ids.max().item())
    if vmin < 0 or vmax >= tokenizer.vocab_size:
        raise ValueError(f"Bad token ids: min={vmin} max={vmax} vocab={tokenizer.vocab_size}")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    logger.info("Running ESMFold model...")
    t0 = time.time()
    out = model(**inputs)
    dt = time.time() - t0

    logger.info(f"Model run complete in {dt:.1f} seconds.")
    pdb_str = model.output_to_pdb(out)[0]

    # pLDDT (optional)
    plddt = None
    for key in ["plddt", "predicted_lddt", "lddt"]:
        if hasattr(out, key):
            plddt = getattr(out, key)
            break
        if isinstance(out, dict) and key in out:
            plddt = out[key]
            break
    mean_plddt = float(plddt.mean().item()) if plddt is not None else None

    return seq, pdb_str, mean_plddt, dt

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

    # node features: (N, 20) one-hot
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

    # 1) sequence edges (type 0): consecutive residues within same chain, in PDB order
    for i in range(N - 1):
        if chains[i] == chains[i + 1]:
            edges.append((i, i + 1)); etypes.append(0)
            edges.append((i + 1, i)); etypes.append(0)

    # 2) spatial edges (type 1): CA distance < cutoff
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

def main():
    start_time = time.time()
    # 换成你要测试的单条序列（CDR3 或 peptide）
    # SEQ = "CASSIRSSYEQYF"   # 示例：TCR beta CDR3 常见长度
    SEQ = "GILGFVFTL"     # 示例：短肽 epitope
    # SEQ = "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYTPGPGIRYPLKDVKELGADVVVVDSGDGVTHVVPIYEGYALPHAILR"
    logger.info(f"Folding single sequence: len={len(SEQ)} seq='{SEQ}'")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # dtype = "float16" if (device == "cuda") else "float32"
    dtype = "float32"  # for CPU, use float32

    logger.info(f"Device={device} dtype={dtype}")
    seq_clean, pdb_str, mean_plddt, seconds = fold_one(SEQ, device=device, dtype=dtype)
    logger.info(f"Folded. seq_len={len(seq_clean)} mean_pLDDT={mean_plddt} time={seconds:.1f}s")

    # Save PDB
    with open("single_output.pdb", "w") as f:
        f.write(pdb_str)
    logger.info("Wrote PDB: single_output.pdb")

    # Build graph
    node_feat, coords, edge_index, edge_type, res_id = build_graph(pdb_str, distance_cutoff=8.0)
    logger.info(f"Graph: N={node_feat.shape[0]} E={edge_index.shape[1]}")
    logger.info(f"edge_type counts: seq={(edge_type==0).sum()} spatial={(edge_type==1).sum()}")

    # Save graph
    np.savez_compressed(
        "single_output_graph.npz",
        node_feat=node_feat,
        coords=coords,
        edge_index=edge_index,
        edge_type=edge_type,
        res_id=res_id
    )
    logger.info("Wrote graph: single_output_graph.npz")
    total_time = time.time() - start_time
    logger.info(f"Total time: {total_time:.1f}s")

if __name__ == "__main__":
    main()
