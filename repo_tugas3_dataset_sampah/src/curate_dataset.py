"""
Pipeline Kurasi Dataset Kustom: Klasifikasi Sampah Daur Ulang
Mata Kuliah: Computer Vision and Deep Learning (Pertemuan 6)

Struktur input (raw, belum di-split):
    dataset_raw/
        organik/            *.jpg
        anorganik_plastik/  *.jpg
        anorganik_logam/    *.jpg

Alur (urutan penting untuk mencegah data leakage):
    1. Audit citra rusak (corrupt)
    2. Deduplikasi (MD5 exact + perceptual hash opsional)
    3. Statistik kelas & deteksi class imbalance
    4. Split stratified 70/15/15 DETERMINISTIK (SEBELUM augmentasi)
    5. Augmentasi dinamis HANYA untuk train
    6. DataLoader (+ WeightedRandomSampler / class weights)
    7. Audit bias visual (statistik kecerahan & warna latar per kelas)
    8. Ekspor Data Card (Markdown)

Jalankan:
    python curate_dataset.py --raw dataset_raw --out dataset_klasifikasi
"""
import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ util
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(root):
    out = []
    for sub, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                out.append(os.path.join(sub, f))
    return sorted(out)


# ------------------------------------------------- 1. audit citra rusak
def audit_corrupt(root, delete=False, quarantine="_corrupt"):
    """Cek integritas. Default: pindahkan ke folder karantina (aman),
    bukan langsung menghapus."""
    checked, bad = 0, []
    for path in list_images(root):
        if quarantine in path:
            continue
        checked += 1
        try:
            with Image.open(path) as im:
                im.verify()              # cek struktur byte
            with Image.open(path) as im:
                im.convert("RGB").load() # verify() saja tidak menangkap semua kasus truncated
        except Exception as e:
            bad.append((path, str(e)))
    for path, err in bad:
        print(f"[RUSAK] {path} ({err})")
        if delete:
            os.remove(path)
        else:
            qdir = os.path.join(root, quarantine)
            os.makedirs(qdir, exist_ok=True)
            shutil.move(path, os.path.join(qdir, os.path.basename(path)))
    print(f"[Audit] diperiksa={checked}, rusak={len(bad)}")
    return len(bad)


# ------------------------------------------------------ 2. deduplikasi
def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path, size=8):
    """Perceptual hash sederhana (difference hash) untuk near-duplicate."""
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def deduplicate(root, near_threshold=4, remove=True):
    seen, dup_exact = {}, []
    for p in list_images(root):
        if "_corrupt" in p or "_duplicates" in p:
            continue
        h = md5_file(p)
        if h in seen:
            dup_exact.append((p, seen[h]))
        else:
            seen[h] = p
    # near-duplicate (hasil foto burst / crop kecil) lintas kelas ikut tertangkap
    uniq = [p for p in seen.values()]
    hashes = {p: dhash(p) for p in uniq}
    dup_near, kept = [], []
    for p in uniq:
        if any(hamming(hashes[p], hashes[q]) <= near_threshold for q in kept):
            dup_near.append(p)
        else:
            kept.append(p)
    print(f"[Dedup] duplikat identik={len(dup_exact)}, mirip (dHash<= {near_threshold})={len(dup_near)}")
    if remove:
        ddir = os.path.join(root, "_duplicates")
        os.makedirs(ddir, exist_ok=True)
        for p, _ in dup_exact:
            shutil.move(p, os.path.join(ddir, f"{md5_file(p)[:8]}_{os.path.basename(p)}"))
        # near-duplicate hanya dilaporkan: periksa manual sebelum dibuang
        with open(os.path.join(ddir, "near_duplicates.txt"), "w") as f:
            f.write("\n".join(dup_near))
    return dup_exact, dup_near


# -------------------------------------------- 3. statistik & imbalance
def class_stats(root):
    ds = datasets.ImageFolder(root, is_valid_file=lambda p: p.lower().endswith(IMG_EXT)
                              and "_corrupt" not in p and "_duplicates" not in p)
    cnt = Counter(ds.targets)
    stats = {ds.classes[k]: v for k, v in sorted(cnt.items())}
    mx, mn = max(stats.values()), min(stats.values())
    print(f"[Kelas] {stats} | rasio imbalance = {mx / mn:.2f}:1")
    return ds, stats


# ------------------------------------------ 4. split stratified & seed
def stratified_split(targets, ratios=(0.70, 0.15, 0.15), seed=SEED):
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for i, t in enumerate(targets):
        by_cls[t].append(i)
    tr, va, te = [], [], []
    for t, idx in by_cls.items():
        rng.shuffle(idx)
        n = len(idx)
        n_tr, n_va = int(round(ratios[0] * n)), int(round(ratios[1] * n))
        tr += idx[:n_tr]
        va += idx[n_tr:n_tr + n_va]
        te += idx[n_tr + n_va:]
    return tr, va, te


def export_split(ds, splits, out_dir):
    """Salin file ke struktur train/val/test (ImageFolder). File asli utuh."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    manifest = {}
    for name, idxs in zip(("train", "val", "test"), splits):
        for i in idxs:
            path, y = ds.samples[i]
            cls = ds.classes[y]
            dst = os.path.join(out_dir, name, cls)
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(path, dst)
            manifest[os.path.basename(path)] = name
    with open(os.path.join(out_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print({k: len(v) for k, v in zip(("train", "val", "test"), splits)})


# ------------------------------------ 5. transformasi (train vs eval)
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),   # aman: sampah tidak punya arah kiri/kanan
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # hue TIDAK diubah
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ------------------------------------------------- 6. DataLoader
def build_loaders(split_dir, use_sampler=True):
    tr = datasets.ImageFolder(os.path.join(split_dir, "train"), transform=train_tf)
    va = datasets.ImageFolder(os.path.join(split_dir, "val"), transform=eval_tf)
    te = datasets.ImageFolder(os.path.join(split_dir, "test"), transform=eval_tf)

    counts = np.bincount(tr.targets, minlength=len(tr.classes))
    class_w = counts.sum() / (len(counts) * counts)          # w_c = N / (C * n_c)
    print(f"[Imbalance] n per kelas (train)={counts.tolist()}, bobot={np.round(class_w, 3).tolist()}")

    g = torch.Generator().manual_seed(SEED)
    if use_sampler:
        sample_w = [class_w[t] for t in tr.targets]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True, generator=g)
        tl = DataLoader(tr, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    else:
        tl = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, generator=g)
    vl = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    el = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Alternatif WeightedRandomSampler: weighted loss (JANGAN dipakai bersamaan secara berlebihan)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32))
    return tl, vl, el, tr.classes, criterion


# ------------------------------------------------ 7. audit bias visual
def bias_audit(root, max_per_class=200):
    """Hitung rata-rata kecerahan & warna rata-rata per kelas.
    Jika satu kelas jauh berbeda dari kelas lain, curigai shortcut learning."""
    ds = datasets.ImageFolder(root)
    rows = defaultdict(list)
    for path, y in ds.samples[:]:
        if len(rows[y]) >= max_per_class:
            continue
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
        border = np.concatenate([a[:6].reshape(-1, 3), a[-6:].reshape(-1, 3),
                                 a[:, :6].reshape(-1, 3), a[:, -6:].reshape(-1, 3)])
        rows[y].append([a.mean(), *a.mean((0, 1)), *border.mean(0)])
    print("\n[Bias] rerata per kelas  (brightness | R G B | border R G B)")
    report = {}
    for y, v in sorted(rows.items()):
        m = np.mean(v, 0)
        report[ds.classes[y]] = m.round(3).tolist()
        print(f"  {ds.classes[y]:20s} {m[0]:.3f} | {m[1]:.3f} {m[2]:.3f} {m[3]:.3f} | "
              f"{m[4]:.3f} {m[5]:.3f} {m[6]:.3f}")
    allv = np.array(list(report.values()))
    spread = allv[:, 0].max() - allv[:, 0].min()
    if spread > 0.10:
        print(f"  [PERINGATAN] selisih kecerahan antar kelas = {spread:.2f} (>0.10): "
              "indikasi bias pencahayaan. Seragamkan pengambilan foto.")
    return report


def background_ablation(model, loader, device="cpu"):
    """Uji shortcut learning: bandingkan akurasi citra asli vs citra yang
    bagian tengahnya (objek) ditutup abu-abu. Jika akurasi tetap tinggi
    padahal objek ditutup, model memakai latar sebagai jalan pintas."""
    model.eval().to(device)
    ok_orig = ok_mask = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            ok_orig += (model(x).argmax(1) == y).sum().item()
            xm = x.clone()
            h, w = x.shape[-2:]
            xm[:, :, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.0   # tutup objek (nilai rerata setelah normalisasi)
            ok_mask += (model(xm).argmax(1) == y).sum().item()
            n += y.numel()
    a, b = ok_orig / n, ok_mask / n
    print(f"[Ablasi] akurasi asli={a:.3f}, objek ditutup={b:.3f}")
    if b > 0.7 * a:
        print("  [PERINGATAN] akurasi bertahan walau objek ditutup -> kemungkinan shortcut learning (latar).")
    return a, b


# ---------------------------------------------------- 8. data card
def write_data_card(stats, out_path="DATA_CARD.md"):
    total = sum(stats.values())
    lines = [
        "# LEMBAR DATASET (DATA CARD) - Klasifikasi Sampah Daur Ulang", "",
        "## 1. Motivasi", "- Pertanyaan riset: [isi]", "- Pengumpul/pendana: [isi]", "",
        "## 2. Komposisi",
        f"- Total citra: {total}",
        *[f"- {k}: {v} citra" for k, v in stats.items()],
        "- Rentang resolusi asli: [isi]", "",
        "## 3. Proses Pengumpulan", "- Perangkat: [isi]", "- Waktu & kondisi: [isi]", "",
        "## 4. Pra-pengolahan & Anotasi",
        "- Label: folder per kelas (ImageFolder), diverifikasi manual",
        "- Pembersihan: audit corrupt, dedup MD5 + dHash",
        f"- Split stratified 70/15/15, seed={SEED}, dilakukan SEBELUM augmentasi", "",
        "## 5. Keterbatasan & Potensi Bias", "- [isi]",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[DataCard] ditulis ke {out_path}")


# ------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="dataset_raw")
    ap.add_argument("--out", default="dataset_klasifikasi")
    ap.add_argument("--delete-corrupt", action="store_true")
    args = ap.parse_args()

    set_seed(SEED)
    if not os.path.isdir(args.raw):
        raise SystemExit(f"Folder '{args.raw}' belum ada. Buat subfolder per kelas dulu.")

    audit_corrupt(args.raw, delete=args.delete_corrupt)
    deduplicate(args.raw)
    ds, stats = class_stats(args.raw)
    splits = stratified_split(ds.targets)
    export_split(ds, splits, args.out)
    train_l, val_l, test_l, classes, crit = build_loaders(args.out)
    print("Kelas:", classes)
    bias_audit(os.path.join(args.out, "train"))
    write_data_card(stats)
    print("Selesai. Lanjutkan ke Pertemuan 7 (pelatihan CNN).")
