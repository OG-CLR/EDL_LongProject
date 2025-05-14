import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from resnet import ResNet18
import pandas as pd

def compute_score(ps, pu, qw, qa, w, f):
    ref_w, ref_f = 5.6e6, 2.8e8
    mem  = (1 - (ps + pu)) * (qw/32) * (w/ref_w)
    comp = (1 - ps) * (max(qw,qa)/32) * (f/ref_f)
    return mem + comp

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            _, preds = out.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

def main():
    prune_amounts    = [0.9]
    fine_tune_epochs = 5
    batch_size       = 64
    device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_device       = torch.device('cpu')

    normalize = transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    train_ds = CIFAR10('./data', train=True,  download=True, transform=transform_train)
    test_ds  = CIFAR10('./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

    results = []
    for p in prune_amounts:
        # 1) Load & prune
        model = ResNet18().to(device)
        model.load_state_dict(torch.load("best_baseline_resnet18.pth", map_location=device))
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(m, name='weight', amount=p)

        # 2) Fine-tune
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        model.train()
        for _ in range(fine_tune_epochs):
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                opt.zero_grad()
                loss = crit(model(imgs), lbls)
                loss.backward()
                opt.step()

        # 3) Remove pruning wrappers
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                prune.remove(m, 'weight')

        # 4) Évaluation FP32
        acc32 = evaluate(model, test_loader, device)
        # calcul du score FP32 (qw=32, qa=32)
        w     = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f_mac = 2.8e8
        score32 = compute_score(ps=0.0, pu=p, qw=32, qa=32, w=w, f=f_mac)

        # 5) Quantization FP16 (half) du modèle pruné
        model16 = ResNet18().to(device).half()
        model16.load_state_dict(model.state_dict(), strict=True)

        # 6) Évaluation FP16 (inputs aussi en half)
        acc16 = evaluate(model16, test_loader, device)
        # calcul du score FP16 (qw=16, qa=16)
        score16 = compute_score(ps=0.0, pu=p, qw=16, qa=16, w=w, f=f_mac)

        results.append({
            'prune_amount': p,
            'acc_fp32':     acc32,
            'score_fp32':   score32,
            'acc_fp16':     acc16,
            'score_fp16':   score16,
        })
        print(
            f"p={p:.2f}  "
            f"FP32→acc:{acc32:.2f}%, score:{score32:.4f}  |  "
            f"FP16→acc:{acc16:.2f}%, score:{score16:.4f}"
        )

    # 7) Sauvegarde des résultats
    df = pd.DataFrame(results)
    df.to_csv('prune_fp16_results.csv', index=False)
    print("Saved prune_fp16_results.csv")

if __name__ == '__main__':
    main()