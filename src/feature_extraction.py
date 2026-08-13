"""
feature_extraction.py
---------------------
Turn one collagen image into a row of numbers, and do that for every tumor core.

Plain-language idea:
  An image is just a grid of numbers (how much collagen at each spot).
  This file (1) cleans up those numbers, (2) measures a bunch of things about
  the collagen, and (3) repeats for every core and saves one big spreadsheet.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import skimage as ski


# ----------------------------------------------------------------------
# STEP 1: PREPROCESSING  ("flatten out the lopsided numbers")
# ----------------------------------------------------------------------
def preprocess(image, cofactor=1.0, censor_percentile=99.0):
    """
    Take the raw collagen image and make it easier to work with.

    Two things happen here:
      1. Censor: the brightest few pixels are wild outliers. We cap them at
         the 99th-percentile value so they stop bullying everything else.
         (This is the step the teammate already did.)
      2. arcsinh transform: this is the NEW part. It squishes the number
         scale so huge values come closer to small ones -- like how decibels
         or earthquake magnitudes compress a giant range into a readable one.
         'cofactor' controls how hard we squish; 1.0 is standard for this
         kind of ion-count data.
    """
    # 1. censor the extreme top tail
    cap = np.percentile(image, censor_percentile)
    censored = np.clip(image, 0, cap)

    # 2. arcsinh squish (arcsinh(0)=0, so all the empty pixels stay 0)
    transformed = np.arcsinh(censored / cofactor)
    return transformed


# ----------------------------------------------------------------------
# STEP 2: THE COLLAGEN MASK  ("draw the line between collagen and empty")
# ----------------------------------------------------------------------
def collagen_mask(image):
    """
    Pick a brightness cutoff automatically (triangle method, which suits
    images that are mostly empty background). Everything brighter than the
    cutoff is called collagen -> gives us a black-and-white stencil.
    """
    threshold = ski.filters.threshold_triangle(image)
    return image > threshold


# ----------------------------------------------------------------------
# STEP 3: GABOR TEXTURE  ("is it streaky, and which way do streaks run?")
# ----------------------------------------------------------------------
def _gabor_response(image, kernel):
    """One Gabor 'stamp' slid across the image -> a match-score map."""
    mask = image > 0                       # ignore empty background
    vals = image[mask]
    norm = np.zeros_like(image, dtype=float)
    norm[mask] = (vals - vals.mean()) / (vals.std() + 1e-8)
    real = ndi.convolve(norm, np.real(kernel), mode="reflect")
    imag = ndi.convolve(norm, np.imag(kernel), mode="reflect")
    resp = np.hypot(real, imag)            # combine so we ignore stripe phase
    resp[~mask] = np.nan                   # blank out background
    return resp


def gabor_features(image):
    """
    Try the stripe-stamp at 4 angles and 3 stripe-widths (12 stamps total).
    Each stamp gives a whole response image; we boil it down to 2 numbers
    (average match + how much it varies) so it fits in the spreadsheet.
    -> 12 stamps x 2 numbers = 24 features.
    """
    thetas = np.deg2rad([0, 45, 90, 135])
    freqs = [0.04, 0.08, 0.12]
    feats = {}
    for theta in thetas:
        for f in freqs:
            k = ski.filters.gabor_kernel(frequency=f, theta=theta,
                                         sigma_x=4, sigma_y=4)
            r = _gabor_response(image, k)
            tag = f"gabor_t{int(np.degrees(theta))}_f{f}"
            feats[tag + "_mean"] = float(np.nanmean(r))
            feats[tag + "_var"] = float(np.nanvar(r))
    return feats


# ----------------------------------------------------------------------
# STEP 4: RIDGE FILTERS + SKELETON  ("find fibers, measure the network")
# ----------------------------------------------------------------------
def fiber_features(image):
    """
    Meijering & Sato hunt for thin tube-like shapes (fibers). We threshold
    to get a fiber mask, then 'skeletonize' -> thin each fiber to a 1-pixel
    line (like drawing roads as thin lines on a map). Total line length =
    how much connected fiber network there is.
    """
    feats = {}
    for name, ridge_fn in [("meij", ski.filters.meijering),
                           ("sato", ski.filters.sato)]:
        ridge = ridge_fn(image, black_ridges=False)
        mask = ridge > ski.filters.threshold_otsu(ridge)
        skel = ski.morphology.skeletonize(mask)
        feats[f"{name}_fiber_area"] = float(mask.mean())      # how much fiber
        feats[f"{name}_skeleton_len"] = int(skel.sum())       # network length
    return feats


# ----------------------------------------------------------------------
# STEP 5: STRUCTURE TENSOR  ("are fibers lined up or pointing everywhere?")
# ----------------------------------------------------------------------
def alignment_features(image, mask, sigma=3):
    """
    THE key one for your project's question (organized vs disorganized).

    At every spot we work out which way the collagen is 'flowing'. Then:
      - coherence: 1.0 = fibers all point the same way (very organized),
                   0.0 = random tangle (disorganized).
      - dominant orientation: the single overall direction fibers tend to run.
    We report the average coherence over the collagen, so one number per core.
    """
    Axx, Axy, Ayy = ski.feature.structure_tensor(
        image, sigma=sigma, order="rc"
    )
    # eigenvalues of the little 2x2 flow-matrix at each pixel
    l1, l2 = ski.feature.structure_tensor_eigenvalues((Axx, Axy, Ayy))
    # coherence: how much bigger the strong direction is than the weak one
    coherence = ((l1 - l2) / (l1 + l2 + 1e-8)) ** 2

    # overall direction of collagen flow (averaged over the collagen area)
    orient = 0.5 * np.arctan2(2 * Axy, Axx - Ayy)

    m = mask & np.isfinite(coherence)
    return {
        "alignment_coherence": float(np.nanmean(coherence[m])),
        "dominant_orientation_rad": float(np.nanmean(orient[m])),
    }


# ----------------------------------------------------------------------
# PUT IT TOGETHER: one image -> one row of numbers
# ----------------------------------------------------------------------
def extract_features(raw_image):
    """Run every measurement on a single raw collagen image."""
    img = preprocess(raw_image)
    mask = collagen_mask(img)

    row = {}
    # coverage + brightness
    row["area_fraction"] = float(mask.mean() * 100)
    row["collagen_mean"] = float(img[mask].mean())
    row["collagen_median"] = float(np.median(img[mask]))
    row["collagen_sd"] = float(img[mask].std())
    row["porosity"] = float(100 - mask.mean() * 100)   # empty space %

    row.update(gabor_features(img))          # texture / direction (24)
    row.update(fiber_features(img))          # fibers + network (4)
    row.update(alignment_features(img, mask))  # organization (2)
    return row


# ----------------------------------------------------------------------
# STEP 6: BATCH OVER EVERY CORE  ("do it for all folders, save one CSV")
# ----------------------------------------------------------------------
def batch(histocat_root, out_csv="collagen_features.csv",
          pattern="*In113*Collagen*.tif*"):
    """
    Walk every tumor-core folder inside 'histocat_root', find its collagen
    file(s), measure it, and stack everything into one spreadsheet.

    histocat_root: the folder that contains all the C01_S02_Bavi_TMA_### folders
    Writes one row per collagen image (folder name + file name are kept as labels).
    """
    rows = []
    core_folders = sorted(
        d for d in glob.glob(os.path.join(histocat_root, "*"))
        if os.path.isdir(d)
    )
    for folder in core_folders:
        core_id = os.path.basename(folder)
        collagen_files = glob.glob(os.path.join(folder, pattern))
        if not collagen_files:
            print(f"  [skip] no collagen file in {core_id}")
            continue
        for f in collagen_files:
            raw = ski.io.imread(f)
            row = {"core_id": core_id, "file": os.path.basename(f)}
            row.update(extract_features(raw))
            rows.append(row)
            print(f"  [ok] {core_id}  area={row['area_fraction']:.1f}%  "
                  f"coherence={row['alignment_coherence']:.3f}")

    df = pd.DataFrame(rows).set_index("core_id")
    df.to_csv(out_csv)
    print(f"\nSaved {len(df)} cores x {df.shape[1]} features -> {out_csv}")
    return df


if __name__ == "__main__":
    # Example (edit the path to point at your 'histocat' folder):
    # batch("/path/to/Bavi/histocat", out_csv="collagen_features.csv")
    pass
