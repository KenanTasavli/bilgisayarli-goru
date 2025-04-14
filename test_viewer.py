#!/usr/bin/env python3
# ---------------------------------------------------------------
# test_video.py – Akıcı oynatma + trackbar + MiniCNN tahmini
# ---------------------------------------------------------------
import cv2, torch, time
from train_1 import MiniCNN, IMAGE_SIZE, DEVICE

# ---------------- 1) AYARLAR -----------------------------------
VIDEO_PATH  = r"D:\Code\Bilgisayarli_Goru\multiclasses\car.mp4"
CKPT        = "fast_multilabel_cnn.pth"
THRESH      = 0.4              # sınıf gösterim eşiği
INFER_EVERY = 3                # kaç karede bir tahmin yapılacak
PLAY_SPEED  = 1.0              # oynatma hızı (1 = normal)

# Sınıf adları (etiket sırasına göre)
CLASS_NAMES = [
    "Fil",
    "Basketball",
    "Electric Scooter",
    "Fruit",
    "Planet",
    "Vehicle",
    "Glass",
    "Cup"
]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------- 2) MODEL -------------------------------------
net = MiniCNN().to(DEVICE)
net.load_state_dict(torch.load(CKPT, map_location=DEVICE))
net.eval()

# ---------------- 3) VIDEO AÇMA -------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f" Cannot open video: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
orig_fps     = cap.get(cv2.CAP_PROP_FPS) or 25
frame_delay  = 1.0 / orig_fps / PLAY_SPEED
delay_ms     = int(frame_delay * 1000)

# ---------------- 4) TRACKBAR ----------------------------------
cur_frame = 0  # anlık kare sayacı

# Trackbar konumu değişince çalışır
def on_trackbar(pos):
    global cur_frame
    cur_frame = pos
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

cv2.namedWindow("video", cv2.WINDOW_NORMAL)
cv2.createTrackbar("pos", "video", 0, total_frames-1, on_trackbar)

# ---------------- 5) DÖNGÜ -------------------------------------
last_probs = None
while True:
    ok, frame = cap.read()
    if not ok: break

    cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Belirli aralıklarda tahmin yap
    if cur_frame % INFER_EVERY == 0:
        img_rs = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            (IMAGE_SIZE, IMAGE_SIZE))
        inp = torch.from_numpy(img_rs).float().permute(2,0,1).unsqueeze(0) / 255.
        with torch.no_grad():
            last_probs = torch.sigmoid(net(inp.to(DEVICE)))[0].cpu().numpy()

    # Son tahmini görsele yaz
    vis = frame.copy()
    if last_probs is not None:
        y = 25
        for cid, p in enumerate(last_probs):
            if p >= THRESH:
                cv2.putText(vis, f"{CLASS_NAMES[cid]} {p:.2f}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,255,0), 2, cv2.LINE_AA)
                y += 30

    # Trackbar'ı güncelle
    cv2.setTrackbarPos("pos", "video", cur_frame)


    cv2.imshow("video", vis)

    # Bekleme süresi (minimum 1 ms)
    if cv2.waitKey(max(1, delay_ms)) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
