#!/usr/bin/env python3
# ===============================================================
# Çok‑etiketli sınıflandırma |  Sadece CUDA, güvenli ayarlar
# ===============================================================
import os, sys, glob, cv2, subprocess, time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------
# 0) CUDA kontrolü – CPU’ya düşmeyi engeller
# ---------------------------------------------------------------
if not torch.cuda.is_available():
    sys.exit("  CUDA bulunamadı! NVIDIA sürücüsü / CUDA kütüphanesi kontrol edin.")

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True   # otomatik en hızlı konfigürasyonu seç

# ---------------------------------------------------------------
# 1) Hiperparametreler
# ---------------------------------------------------------------
DATASET_ROOT = r"D:\Code\Bilgisayarli_Goru\multiclasses\dataset"
NUM_CLASSES  = 8
IMAGE_SIZE   = 128
BATCH_SIZE   = 16
EPOCHS       = 50
LR           = 2e-3

# ---------------------------------------------------------------
# 2) Dataset sınıfı – YOLO formatından multi-label vektör üretir
# ---------------------------------------------------------------
class YoloMultiLabelDS(Dataset):
    def __init__(self, root, split="train"):
        self.imgs = sorted(glob.glob(os.path.join(root, "images", split, "*.jpg")))
        self.lbls = [p.replace("images", "labels").replace(".jpg", ".txt") for p in self.imgs]

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])[..., ::-1]  # BGR → RGB
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = torch.from_numpy(img).float().permute(2,0,1) / 255.0

        target = torch.zeros(NUM_CLASSES)
        if os.path.exists(self.lbls[idx]):
            for line in open(self.lbls[idx]):
                cid = int(line.split()[0])
                if 0 <= cid < NUM_CLASSES:
                    target[cid] = 1.0  # çoklu sınıf varsa hepsi 1 yapılır
        return img, target

# DataLoader oluşturan yardımcı fonksiyon
def loader(split, shuffle):
    ds = YoloMultiLabelDS(DATASET_ROOT, split)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=2, pin_memory=True)

# ---------------------------------------------------------------
# 3) Mini CNN modeli – elle tanımlanmış küçük bir ağ
# ---------------------------------------------------------------
def conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
        nn.ReLU(inplace=True)
    )

class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            conv(3, 16), nn.MaxPool2d(2),     # 128 → 64
            conv(16,32), nn.MaxPool2d(2),     # 64  → 32
            conv(32,64), nn.MaxPool2d(2),     # 32  → 16
            conv(64,128),nn.MaxPool2d(2),     # 16  → 8
            nn.AdaptiveAvgPool2d(1)           # 8   → 1
        )
        self.fc = nn.Linear(128, NUM_CLASSES)  # çıkış katmanı

    def forward(self, x):
        return self.fc(self.feat(x).flatten(1))  # (B,128) → (B,num_classes)

# ---------------------------------------------------------------
# 4) GPU sıcaklık ve bellek bilgisi (isteğe bağlı)
# ---------------------------------------------------------------
def gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            encoding="utf-8")
        temp, mem_used, mem_tot = map(int, out.strip().split(", "))
        return f"GPU {temp}°C | {mem_used}/{mem_tot} MB"
    except Exception:
        return "nvidia-smi erişilemedi"

# ---------------------------------------------------------------
# 5) Eğitim döngüsü
# ---------------------------------------------------------------
def train():
    print("Cihaz:", torch.cuda.get_device_name(0))
    net = MiniCNN().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), LR)
    loss_fn = nn.BCEWithLogitsLoss()           # çoklu etiket için uygun
    scaler = torch.cuda.amp.GradScaler()       # otomatik karma hassasiyet (AMP)

    train_loader = loader("train", True)

    for ep in range(EPOCHS):
        net.train(); tot = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{EPOCHS}")
        for imgs, tgts in pbar:
            imgs, tgts = imgs.to(DEVICE, non_blocking=True), tgts.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast():     # daha hızlı ve daha az VRAM
                loss = loss_fn(net(imgs), tgts)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
            tot += loss.item()
            if len(pbar)%20==0:
                pbar.set_postfix(loss=f"{tot/(len(pbar)):.3f}",
                                 gpu=gpu_stats())  # GPU bilgisi yaz

        print(f"  ↳ epoch loss {tot/len(train_loader):.4f}  |  {gpu_stats()}")

    torch.save(net.state_dict(), "fast_multilabel_cnn.pth")
    print("  Eğitim bitti → fast_multilabel_cnn.pth")

if __name__ == "__main__":
    train()
