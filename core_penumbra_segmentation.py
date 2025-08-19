import os, glob, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=256):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.mask_paths = [os.path.join(mask_dir, os.path.basename(p)) for p in self.img_paths]
        self.size = size
        self.tf = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((size, size)),
            transforms.ToTensor(),                    
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, i):
        img = Image.open(self.img_paths[i]).convert("RGB")
        msk = Image.open(self.mask_paths[i]).convert("L")  # integer labels 0/1/2
        img = self.tf(img)
        msk = msk.resize((self.size, self.size), resample=Image.NEAREST)
        msk = torch.from_numpy(np.array(msk, dtype=np.int64))
        return img, msk


def conv_block(ic, oc):
    return nn.Sequential(
        nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(oc, oc, 3, padding=1), nn.ReLU(inplace=True)
    )

class TinyUNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.e1 = conv_block(1, 16)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = conv_block(16, 32)
        self.p2 = nn.MaxPool2d(2)
        self.b  = conv_block(32, 64)
        self.u2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d2 = conv_block(64, 32)
        self.u1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.d1 = conv_block(32, 16)
        self.head = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        e1 = self.e1(x); p1 = self.p1(e1)
        e2 = self.e2(p1); p2 = self.p2(e2)
        b  = self.b(p2)
        u2 = self.u2(b); d2 = self.d2(torch.cat([u2, e2], 1))
        u1 = self.u1(d2); d1 = self.d1(torch.cat([u1, e1], 1))
        return self.head(d1)  # (B,C,H,W)


@torch.no_grad()
def dice_per_class(logits, target, eps=1e-6):
    pred = logits.argmax(1)
    C = logits.shape[1]
    out = []
    for c in range(C):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p*t).sum()
        denom = p.sum()+t.sum()
        out.append(((2*inter+eps)/(denom+eps)).item())
    return out  # [bg, core, penumbra]

def save_mask(mask_tensor, path):
    # mask_tensor: (H,W) int64 in {0,1,2}
    Image.fromarray(mask_tensor.cpu().numpy().astype(np.uint8)).save(path)


def train(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SegDataset(args.images, args.masks, size=args.size)
    n_val = max(1, int(0.2*len(ds)))
    n_tr  = len(ds)-n_val
    tr_ds, va_ds = torch.utils.data.random_split(ds, [n_tr, n_val])
    tl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    vl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = TinyUNet(3).to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = 0.0
    for ep in range(1, args.epochs+1):
        model.train(); run = 0.0
        pbar = tqdm(tl, desc=f"Epoch {ep}/{args.epochs}", ncols=100)
        for img, msk in pbar:
            img, msk = img.to(device), msk.to(device)
            opt.zero_grad()
            logits = model(img)
            loss = crit(logits, msk)
            loss.backward(); opt.step()
            run += loss.item(); pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval(); dices=[]
        with torch.no_grad():
            for img, msk in vl:
                img, msk = img.to(device), msk.to(device)
                logits = model(img)
                dices.append(dice_per_class(logits, msk))
        if dices:
            d = np.array(dices).mean(0)  # [bg, core, pen]
            fg = d[1:].mean()
            print(f"Val Dice  bg:{d[0]:.3f}  core:{d[1]:.3f}  pen:{d[2]:.3f}")
            if fg > best:
                best = fg
                ck = os.path.join(args.out, "best.pth")
                torch.save(model.state_dict(), ck)
                print(f"Saved {ck}")

    torch.save(model.state_dict(), os.path.join(args.out, "final.pth"))

@torch.no_grad()
def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet(3).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    img = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    logits = model(img)
    pred = logits.argmax(1)[0].cpu()
    save_mask(pred, args.out)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str)
    ap.add_argument("--masks", type=str)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="checkpoints")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--ckpt", type=str, help="checkpoint path for predict")
    ap.add_argument("--image", type=str, help="input image path for predict")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.predict:
        assert args.ckpt and args.image and args.out, "Need --ckpt, --image, --out for prediction"
        predict(args)
    else:
        assert args.images and args.masks, "Need --images and --masks for training"
        train(args)
