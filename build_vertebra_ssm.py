"""
build_vertebra_ssm.py

Per-vertebra Statistical Shape Model (SSM) construction.

Purpose
-------
Given a dataset of segmented spine CT volumes (e.g. VerSe), this script builds
ONE separate SSM per vertebra level (C1..L5, or whichever labels you pass in).
Each SSM = mean shape + PCA modes of variation, in dense point correspondence.

Why per-level and why correspondence matters
---------------------------------------------
PCA-based shape modeling requires every training shape to be represented by
the SAME NUMBER of points, in the SAME ANATOMICAL ORDER (point i on subject A's
T5 must correspond to point i on subject B's T5 -- e.g. both are "top-left
corner of the superior endplate"). Raw segmentation masks don't give you that
for free -- two different people's T5 meshes have different numbers of
vertices in different orders. So the pipeline is:

    1. Extract a surface mesh for one vertebra level from each subject's
       segmentation (marching cubes).
    2. Pick one subject as the TEMPLATE for that level.
    3. Non-rigidly register every other subject's mesh onto the template
       using Coherent Point Drift (CPD). This resamples every subject's
       shape into the template's point ordering/count -> correspondence.
    4. Rigidly align (Procrustes: rotation + translation + optional scale)
       all corresponded shapes into one reference frame, removing pose
       differences so PCA only captures SHAPE variation, not position.
    5. Run PCA over the aligned, corresponded point clouds.
    6. Save mean_shape + top-k principal components + explained variance.

Run this once per vertebra level (loop externally, or use --all_levels).

Expected input layout (VerSe-style)
------------------------------------
data_root/
    sub-001/
        sub-001_seg.nii.gz      <- multi-label mask, one integer label per vertebra
    sub-002/
        sub-002_seg.nii.gz
    ...

VerSe label convention: 1-7 = C1-C7, 8-19 = T1-T12, 20-24 = L1-L5
(23/24/25 sometimes used for L6/sacrum/extra levels depending on version --
check the dataset's label README and adjust VERSE_LABELS below if needed).

Dependencies
------------
pip install nibabel scikit-image numpy scipy scikit-learn open3d pycpd --break-system-packages

Usage
-----
python build_vertebra_ssm.py \
    --data_root /path/to/verse \
    --level T5 \
    --n_points 1024 \
    --n_modes 15 \
    --out_dir ./ssm_models
"""

import argparse
import glob
import os
import pickle
from dataclasses import dataclass

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from skimage.measure import marching_cubes
except ImportError:
    marching_cubes = None

try:
    from pycpd import DeformableRegistration
except ImportError:
    DeformableRegistration = None

from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# VerSe label convention (adjust if your dataset version differs)
# ---------------------------------------------------------------------------
VERSE_LABELS = {
    **{f"C{i}": i for i in range(1, 8)},        # C1-C7   -> 1-7
    **{f"T{i}": i + 7 for i in range(1, 13)},    # T1-T12  -> 8-19
    **{f"L{i}": i + 19 for i in range(1, 6)},    # L1-L5   -> 20-24
}


@dataclass
class SSM:
    level: str
    mean_shape: np.ndarray          # (n_points, 3)
    components: np.ndarray          # (n_modes, n_points*3)
    explained_variance: np.ndarray  # (n_modes,)
    explained_variance_ratio: np.ndarray
    n_points: int
    n_training_subjects: int


# ---------------------------------------------------------------------------
# Step A: extract a single vertebra's surface mesh from a segmentation volume
# ---------------------------------------------------------------------------
def extract_vertebra_points(seg_path: str, label_value: int, n_sample: int = 4096) -> np.ndarray:
    """
    Load a segmentation NIfTI, isolate one vertebra label, run marching cubes
    to get a surface mesh, and return a point cloud (subsampled vertices) in
    physical (mm) coordinates using the image affine.
    """
    if nib is None or marching_cubes is None:
        raise ImportError("Requires nibabel and scikit-image: "
                           "pip install nibabel scikit-image --break-system-packages")

    img = nib.load(seg_path)
    data = img.get_fdata()
    affine = img.affine

    mask = (data == label_value).astype(np.uint8)
    if mask.sum() == 0:
        raise ValueError(f"Label {label_value} not found in {seg_path}")

    # Marching cubes needs a bit of padding so the surface closes at the volume edge
    padded = np.pad(mask, 1, mode="constant")
    verts, faces, _, _ = marching_cubes(padded, level=0.5)
    verts -= 1  # undo padding offset

    # voxel -> world/physical coordinates (mm), so subjects of different voxel
    # sizes/resolutions become comparable
    verts_h = np.hstack([verts, np.ones((verts.shape[0], 1))])
    verts_world = (affine @ verts_h.T).T[:, :3]

    # Subsample for tractable CPD (CPD is O(n^2)-ish); random subset is fine,
    # since CPD will re-establish correspondence to the template regardless.
    if verts_world.shape[0] > n_sample:
        idx = np.random.choice(verts_world.shape[0], n_sample, replace=False)
        verts_world = verts_world[idx]

    return verts_world


# ---------------------------------------------------------------------------
# Step B: non-rigid correspondence via Coherent Point Drift onto a template
# ---------------------------------------------------------------------------
def register_to_template(source_points: np.ndarray, template_points: np.ndarray) -> np.ndarray:
    """
    Deform `source_points` onto `template_points` using CPD so the OUTPUT has
    exactly len(template_points) points, ordered to correspond to the
    template's points. This is what makes cross-subject PCA valid.
    """
    if DeformableRegistration is None:
        raise ImportError("Requires pycpd: pip install pycpd --break-system-packages")

    reg = DeformableRegistration(X=template_points, Y=source_points, alpha=2.0, beta=2.0)
    transformed_source, _ = reg.register()
    # transformed_source now has the same number of points as template_points,
    # each index i approximately corresponding to template point i.
    return transformed_source


# ---------------------------------------------------------------------------
# Step C: rigid (Procrustes) alignment to remove pose, keep shape only
# ---------------------------------------------------------------------------
def procrustes_align(points: np.ndarray, reference: np.ndarray, allow_scale: bool = False):
    """
    Rigid (+ optional uniform scale) alignment of `points` onto `reference`.
    Removes translation/rotation(/scale) so PCA captures shape, not pose.
    """
    mu_p, mu_r = points.mean(axis=0), reference.mean(axis=0)
    p0, r0 = points - mu_p, reference - mu_r

    if allow_scale:
        norm_p = np.linalg.norm(p0)
        norm_r = np.linalg.norm(r0)
        p0 /= norm_p
        r0 /= norm_r

    # Kabsch algorithm for optimal rotation
    U, _, Vt = np.linalg.svd(p0.T @ r0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    aligned = (R @ p0.T).T
    if allow_scale:
        aligned *= norm_r
    return aligned + mu_r


# ---------------------------------------------------------------------------
# Step D: build the SSM for one vertebra level across all subjects
# ---------------------------------------------------------------------------
def build_ssm_for_level(data_root: str, level: str, n_points: int, n_modes: int) -> SSM:
    label_value = VERSE_LABELS[level]
    seg_paths = sorted(glob.glob(os.path.join(data_root, "sub-*", "*_seg.nii.gz")))
    if not seg_paths:
        raise FileNotFoundError(f"No segmentation files found under {data_root}")

    print(f"[{level}] Found {len(seg_paths)} candidate subjects")

    # 1. Extract raw point clouds for this level from every subject that has it
    raw_clouds = []
    for path in seg_paths:
        try:
            pts = extract_vertebra_points(path, label_value, n_sample=n_points * 4)
            raw_clouds.append(pts)
        except ValueError:
            continue  # this subject doesn't have this level segmented -- skip

    if len(raw_clouds) < 5:
        raise RuntimeError(
            f"[{level}] Only {len(raw_clouds)} usable subjects found -- "
            "need more for a meaningful SSM (recommend 20+)."
        )
    print(f"[{level}] {len(raw_clouds)} subjects have this level segmented")

    # 2. Pick the subject with the median point count as the template
    #    (avoids an unusually noisy/incomplete segmentation becoming the template)
    counts = [c.shape[0] for c in raw_clouds]
    template_idx = int(np.argsort(counts)[len(counts) // 2])
    template = raw_clouds[template_idx]
    if template.shape[0] > n_points:
        idx = np.random.choice(template.shape[0], n_points, replace=False)
        template = template[idx]
    print(f"[{level}] Using subject #{template_idx} as template ({n_points} pts)")

    # 3. Register every subject's cloud onto the template -> correspondence
    corresponded = []
    for i, cloud in enumerate(raw_clouds):
        if i == template_idx:
            corresponded.append(template)
            continue
        try:
            matched = register_to_template(cloud, template)
            corresponded.append(matched)
            print(f"[{level}]   registered subject {i+1}/{len(raw_clouds)}")
        except Exception as e:
            print(f"[{level}]   subject {i} failed registration, skipping: {e}")

    # 4. Rigid-align all corresponded shapes to remove pose differences
    ref = corresponded[0]
    aligned = [procrustes_align(pts, ref) for pts in corresponded]
    aligned = np.stack(aligned, axis=0)  # (n_subjects, n_points, 3)

    # 5. PCA over flattened shape vectors
    n_subjects = aligned.shape[0]
    X = aligned.reshape(n_subjects, -1)  # (n_subjects, n_points*3)
    mean_shape = X.mean(axis=0)

    n_modes_eff = min(n_modes, n_subjects - 1)
    pca = PCA(n_components=n_modes_eff)
    pca.fit(X - mean_shape)

    ssm = SSM(
        level=level,
        mean_shape=mean_shape.reshape(-1, 3),
        components=pca.components_,               # (n_modes, n_points*3)
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        n_points=n_points,
        n_training_subjects=n_subjects,
    )
    print(f"[{level}] SSM built: {n_modes_eff} modes explain "
          f"{ssm.explained_variance_ratio.sum()*100:.1f}% of shape variance")
    return ssm


def save_ssm(ssm: SSM, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ssm_{ssm.level}.pkl")
    with open(path, "wb") as f:
        pickle.dump(ssm, f)
    print(f"[{ssm.level}] Saved -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Build per-vertebra SSMs from VerSe-style data")
    parser.add_argument("--data_root", required=True, help="Root folder containing sub-XXX subject folders")
    parser.add_argument("--level", default=None, help="Single level to build, e.g. T5. Omit if using --all_levels")
    parser.add_argument("--all_levels", action="store_true", help="Build SSMs for all 24 levels (C1-L5)")
    parser.add_argument("--n_points", type=int, default=1024, help="Number of corresponded points per shape")
    parser.add_argument("--n_modes", type=int, default=15, help="Number of PCA modes to keep")
    parser.add_argument("--out_dir", default="./ssm_models")
    args = parser.parse_args()

    levels = list(VERSE_LABELS.keys()) if args.all_levels else [args.level]
    if not args.all_levels and args.level is None:
        parser.error("Provide --level or --all_levels")

    for level in levels:
        try:
            ssm = build_ssm_for_level(args.data_root, level, args.n_points, args.n_modes)
            save_ssm(ssm, args.out_dir)
        except Exception as e:
            print(f"[{level}] SKIPPED: {e}")


if __name__ == "__main__":
    main()
