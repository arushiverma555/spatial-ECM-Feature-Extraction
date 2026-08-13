"""
Larynx ECM feature extraction — batch over the histocat folder structure.

Preprocessing/threshold aligned with Blake's normal_correlation_matrix.ipynb
so both pipelines produce comparable feature values:
  * preprocess = clip at 99th percentile + linear rescale to [0, 1]
  * threshold  = triangle threshold on the full scaled image
Correlation downstream = Pearson.

Folder layout (one folder per core):
  histocat/
    20251215 Larynx TMA Section 1_001/
        In113_113In_Collagen.tiff        <- collagen channel (what we use)
        Nd143_143Nd_VIM.tiff             <- other markers (ignored)
        ...

Joins each core to status_key.csv (TMA_ID, Image, Status), keeps normal cores,
writes a feature table with TMA_ID as the first column.

USAGE (edit the paths near the bottom, then run):
    python larynx_batch.py
"""
import os
import numpy as np
import pandas as pd
import tifffile
import skimage as ski
from scipy import ndimage as ndi

COLLAGEN_FILE = "In113_113In_Collagen.tiff"   # the channel we analyze

# ---------------- preprocessing (matches Blake) ----------------
def preprocess(img):
    """Clip at the 99th percentile, then linearly rescale to [0, 1]."""
    img = img.astype(float)
    p99 = np.percentile(img, 99)
    censored = np.clip(img, 0, p99)
    if p99 <= 0:
        return np.zeros_like(img)
    return ski.exposure.rescale_intensity(censored, in_range=(0, p99), out_range=(0, 1))

def collagen_mask(scaled):
    """Triangle threshold on the full scaled image (matches Blake).
    NOTE: because ~76% of pixels are zero, this cutoff usually lands below the
    smallest nonzero value, so the mask ~= all nonzero pixels and area_fraction
    ~= nonzero fraction. Kept this way on purpose so our two pipelines agree."""
    thr = ski.filters.threshold_triangle(scaled)
    return scaled > thr

# ---------------- feature extraction ----------------
def gabor_features(scaled):
    """4 orientations x 3 frequencies; mean & variance of each response -> 24 features."""
    feats = {}
    m = scaled > 0
    if m.sum() == 0:
        for th in (0,45,90,135):
            for f in (0.04,0.08,0.12):
                feats[f"gabor_t{th}_f{f}_mean"]=np.nan; feats[f"gabor_t{th}_f{f}_var"]=np.nan
        return feats
    v = scaled[m]; norm = np.zeros_like(scaled); norm[m] = (v - v.mean())/(v.std()+1e-8)
    for th in (0,45,90,135):
        for f in (0.04,0.08,0.12):
            k = ski.filters.gabor_kernel(frequency=f, theta=np.deg2rad(th), sigma_x=4, sigma_y=4)
            r = ndi.convolve(norm, np.real(k), mode="reflect")
            i = ndi.convolve(norm, np.imag(k), mode="reflect")
            resp = np.hypot(r, i)[m]
            feats[f"gabor_t{th}_f{f}_mean"] = float(resp.mean())
            feats[f"gabor_t{th}_f{f}_var"]  = float(resp.var())
    return feats

def fiber_features(scaled):
    """Ridge detection (Meijering + Sato) + skeleton network length & fiber area."""
    feats = {}
    for name, filt in (("meij", ski.filters.meijering), ("sato", ski.filters.sato)):
        ridge = filt(scaled, black_ridges=False)
        try:
            mm = ridge > ski.filters.threshold_otsu(ridge)
        except Exception:
            mm = ridge > 0
        clean = ski.morphology.remove_small_objects(mm, min_size=50)
        skel = ski.morphology.skeletonize(clean)
        feats[f"{name}_fiber_area"]   = float(clean.mean())
        feats[f"{name}_skeleton_len"] = float(skel.sum())
    return feats

def alignment_features(scaled, mask):
    """Structure tensor -> mean coherence (alignment) and dominant orientation."""
    Axx, Axy, Ayy = ski.feature.structure_tensor(scaled, sigma=3, order="rc")
    l1, l2 = ski.feature.structure_tensor_eigenvalues((Axx, Axy, Ayy))
    coh = (l1 - l2) / (l1 + l2 + 1e-8)
    orient = 0.5 * np.arctan2(2*Axy, Axx - Ayy)
    m = mask if mask.sum() > 0 else np.ones_like(mask, dtype=bool)
    return {
        "alignment_coherence": float(coh[m].mean()),
        "dominant_orientation_rad": float(np.median(orient[m])),
    }

def extract_features(collagen_path):
    img = tifffile.imread(collagen_path).astype(float)
    scaled = preprocess(img)
    mask = collagen_mask(scaled)
    feats = {}
    feats["area_fraction"] = float(mask.mean() * 100)          # % collagen (density)
    feats["porosity"]      = float(1 - mask.mean())
    if mask.sum() > 0:
        feats["collagen_mean"]   = float(scaled[mask].mean())
        feats["collagen_median"] = float(np.median(scaled[mask]))
        feats["collagen_sd"]     = float(scaled[mask].std())
    else:
        feats["collagen_mean"]=feats["collagen_median"]=feats["collagen_sd"]=np.nan
    feats.update(gabor_features(scaled))
    feats.update(fiber_features(scaled))
    feats.update(alignment_features(scaled, mask))
    return feats

# ---------------- batch over the histocat structure ----------------
def batch(histocat_dir, status_key_csv, out_full, out_status_filter=("normal",)):
    key = pd.read_csv(status_key_csv)
    key["folder"] = key["Image"].str.replace(r"\.tiff?$", "", regex=True)
    lookup = key.set_index("folder")[["TMA_ID", "Status"]].to_dict("index")

    rows = []
    folders = sorted(d for d in os.listdir(histocat_dir)
                     if os.path.isdir(os.path.join(histocat_dir, d)))
    for folder in folders:
        info = lookup.get(folder)
        if info is None:
            continue                                    # not in our cohort (other sections)
        if out_status_filter and info["Status"] not in out_status_filter:
            continue                                    # keep only requested statuses
        collagen = os.path.join(histocat_dir, folder, COLLAGEN_FILE)
        if not os.path.exists(collagen):
            print(f"  [warn] no collagen file downloaded in: {folder}")
            continue
        print(f"  processing {info['TMA_ID']} ({info['Status']})")
        feats = extract_features(collagen)
        rows.append({"TMA_ID": info["TMA_ID"], "Status": info["Status"], **feats})

    df = pd.DataFrame(rows)
    cols = ["TMA_ID"] + [c for c in df.columns if c != "TMA_ID"]
    df = df[cols]
    df.to_csv(out_full, index=False)
    print(f"\nWrote {len(df)} cores x {df.shape[1]-1} columns -> {out_full}")
    return df

if __name__ == "__main__":
    # ----- paths relative to the ECM project folder -----
    HISTOCAT_DIR   = "histocat"
    STATUS_KEY_CSV = "status_key.csv"
    OUT_FULL       = "Features/features_full_normal.csv"

    batch(HISTOCAT_DIR, STATUS_KEY_CSV, OUT_FULL, out_status_filter=("normal",))