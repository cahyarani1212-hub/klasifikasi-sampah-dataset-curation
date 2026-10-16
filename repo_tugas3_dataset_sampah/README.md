# Klasifikasi Sampah Daur Ulang: Kurasi Dataset & Analisis Bias

Tugas 3 - Computer Vision and Deep Learning (Pertemuan 6: Persiapan Dataset Citra)

- Nama: [isi]
- NIM: [isi]
- Dosen Pengampu: Edwin Hari Agus Prastyo, S.Kom., M.Kom.

## Deskripsi
Pipeline kurasi dataset citra untuk klasifikasi tiga kelas sampah: `organik`, `anorganik_plastik`, `anorganik_logam`.
Pipeline mencakup audit file rusak, deduplikasi (MD5 + dHash), split stratified 70/15/15 **sebelum** augmentasi,
augmentasi dinamis khusus data train, penanganan class imbalance, audit bias visual, dan Data Card.

## Struktur Repositori
```
.
├── README.md
├── requirements.txt
├── DATA_CARD.md                  # dihasilkan/diisi dari hasil pipeline
├── src/
│   └── curate_dataset.py         # pipeline lengkap
├── docs/
│   ├── Laporan_Tugas3_Dataset_Sampah.pdf
│   └── Jawaban_Latihan_Formatif_P6.pdf
└── dataset_raw/                  # taruh foto di sini, satu subfolder per kelas
    ├── organik/
    ├── anorganik_plastik/
    └── anorganik_logam/
```

## Cara Menjalankan
```bash
pip install -r requirements.txt
# 1. Taruh foto ke dataset_raw/<kelas>/
# 2. Jalankan pipeline
python src/curate_dataset.py --raw dataset_raw --out dataset_klasifikasi
```
Keluaran: folder `dataset_klasifikasi/{train,val,test}/<kelas>/` (format ImageFolder), `split_manifest.json`, dan `DATA_CARD.md`.
Dapat dijalankan di Google Colab dengan mengunggah folder `dataset_raw`.

## Keputusan Metodologis
| Isu | Keputusan |
|---|---|
| Data leakage | Split dahulu, augmentasi hanya di train (dinamis, di memori) |
| Reproduktibilitas | Seed 42, split stratified, manifest split disimpan |
| Citra rusak/duplikat | Dipindah ke karantina (`_corrupt`, `_duplicates`), tidak langsung dihapus |
| Augmentasi | Resize, RandomResizedCrop, HorizontalFlip, Rotation 15, ColorJitter tanpa hue |
| Class imbalance | WeightedRandomSampler atau weighted CrossEntropyLoss (pilih salah satu) |
| Bias visual | `bias_audit()` dan `background_ablation()` untuk mendeteksi shortcut learning |

## Dataset
- Sumber: pengumpulan mandiri dengan kamera ponsel [isi]; sumber unduhan (jika ada): [nama, lisensi, tautan]
- Jumlah citra: [isi] (target minimal 30 per kelas, 3 kelas)
- Foto mentah tidak diunggah ke repo (lihat `.gitignore`) [ubah bila diperlukan]. Tautan dataset: [isi bila dibagikan, mis. Google Drive/Kaggle]

## Laporan
Lihat `docs/Laporan_Tugas3_Dataset_Sampah.pdf` (Bab I-IV + Data Card).

## Referensi
- Gebru, T. et al. (2021). Datasheets for Datasets. Communications of the ACM, 64(12), 86-92.
- PyTorch Documentation. Torchvision Transforms and Datasets API.
- Skalski, P. (2024). Make Sense. https://www.makesense.ai/
- NIST (2023). AI Risk Management Framework 1.0.
