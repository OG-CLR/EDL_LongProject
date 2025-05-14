import torch
import torch.nn as nn
import torch.quantization
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from resnet import ResNet18
import pandas as pd
from thop import profile
from ptflops import get_model_complexity_info


def compute_model_score(model, input_size=(3, 32, 32), 
                        ps=0.0, pu=0.0, qw=32, qa=32, 
                        reference_w=5.6e6, reference_f=2.8e8, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calcule le score de compression d'un modèle donné selon la formule :
    score = [(1 - (ps + pu)) * (qw / 32) * (w / ref_w)] + [(1 - ps) * (max(qw, qa) / 32) * (f / ref_f)]
    """
  
    model = model.to(device).eval()

    # Calcul du nombre de paramètres
    w = sum(p.numel() for p in model.parameters())

    # Calcul des MACs (Multiply-Adds)
    with torch.amp.autocast(device_type='cuda', enabled=False):
  # désactiver AMP pour mesures précises
        macs, _ = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
        f = macs  # nombre d’opérations MACs

    # Application de la formule
    score_param = ((1 - (ps + pu)) * (qw / 32) * (w / reference_w))
    score_ops   = ((1 - ps) * (max(qw, qa) / 32) * (f / reference_f))

    total_score = score_param + score_ops
    return total_score






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
    
    score32 = compute_model_score(model_fp32, ps=0.0, pu=0.0, qw=32, qa=32)
    print(f"FP32      → acc: {acc32:.2f}%, score: {score32:.4f}")

    

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
    
    score16 = compute_model_score(model16, ps=0.0, pu=0.0, qw=16, qa=16)
    print(f"FP16      → acc: {acc16:.2f}%, score: {score16:.4f}")
    results.append({'method':'FP16', 'accuracy':acc16, 'score':score16})

    

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('quant_results.csv', index=False)
    print("Saved quant_results.csv")

if __name__ == '__main__':
    main()
