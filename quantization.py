import torch
import torch.nn as nn
import torch.quantization
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from resnet import ResNet18
import pandas as pd
from thop import profile


def compute_score32(ps, pu, qw, qa, w, f):
    # Référence: w_ref=5.6e6/11173962, f_ref=2.8e8/278940160.0
    ref_w, ref_f = 11173962, 278940160.0
    mem = (1 - (ps + pu)) * (qw / 32) * (w / ref_w)
    comp = (1 - ps) * (max(qw, qa) / 32) * (f / ref_f)
    return mem + comp

def compute_score16(ps, pu, qw, qa, w, f):
    # Référence: w_ref=5.6e6/11173962, f_ref=2.8e8/278940160.0
    ref_w, ref_f = 5.6e6, 2.8e8
    mem = (1 - (ps + pu)) * (qw / 32) * (w / ref_w)
    comp = (1 - ps) * (max(qw, qa) / 32) * (f / ref_f)
    return mem + comp

def compute_macs(model, input_size=(1, 3, 32, 32), device='cuda'):
    """
    Retourne le nombre de MACs (mult-adds) pour le modèle donné.
    THOP renvoie le nombre de FLOPs (multiplications + additions),
    on divise donc par 2 pour obtenir le nombre de MACs.
    """
    model = model.to(device)
    dummy = torch.randn(*input_size).to(device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    macs = flops / 2
    return macs

def compute_model_size(model, input_size=(1,3,32,32), device='cuda'):
    # nombre de poids
    w = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # nombre de MACs
    f = compute_macs(model, input_size=input_size, device=device)
    return w, f


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            _, pred = out.max(1)
            correct += pred.eq(lbls).sum().item()
            total += lbls.size(0)
    return 100. * correct / total

def main():
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_device = torch.device('cpu')

    # Data
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010))
    ])
    ds     = CIFAR10('/opt/img/effdl-cifar10/', train=False, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    # 0) Baseline FP32
    model_fp32 = ResNet18().to(device)
    model_fp32.load_state_dict(torch.load("best_baseline_resnet18.pth", map_location=device))
    acc32 = evaluate(model_fp32, loader, device)
    w, f = compute_model_size(model, input_size=(1,3,32,32), device=device)
    score32 = compute_score32(ps=0.0, pu=prune_amount, qw=32, qa=32, w=w, f=f)
    print(f"FP32      → acc: {acc32:.2f}%, score: {score32:.4f}")

    # 0.5) Compute effective qw for dynamic INT8
    lin_params  = sum(m.weight.numel() for m in model_fp32.modules() if isinstance(m, nn.Linear))
    conv_params = sum(m.weight.numel() for m in model_fp32.modules() if isinstance(m, nn.Conv2d))
    total_params = lin_params + conv_params
    qw_eff = (8 * lin_params + 32 * conv_params) / total_params
    # qw_eff will be <32, reflecting that only linears are 8-bit

    results = [{'method':'FP32', 'accuracy':acc32, 'score':score32}]

    # 1) FP16 quantization
    model16 = ResNet18().to(device).half()
    model16.load_state_dict(model_fp32.state_dict(), strict=True)
    loader16 = DataLoader(
        ds, batch_size=64, shuffle=False,
        collate_fn=lambda batch: (
            torch.stack([b[0].half() for b in batch]),
            torch.tensor([b[1] for b in batch])
        )
    )
    acc16 = evaluate(model16, loader16, device)
    w, f = compute_model_size(model, input_size=(1,3,32,32), device=device)
    score16 = compute_score16(ps=0.0, pu=prune_amount, qw=16, qa=16, w=w, f=f)
    print(f"FP16      → acc: {acc16:.2f}%, score: {score16:.4f}")
    results.append({'method':'FP16', 'accuracy':acc16, 'score':score16})

    

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('quant_results.csv', index=False)
    print("Saved quant_results.csv")

if __name__ == '__main__':
    main()
