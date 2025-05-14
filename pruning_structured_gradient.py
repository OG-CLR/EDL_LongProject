import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
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






def main():
    # Hyperparamètres
    prune_amounts    = [0.1, 0.2]
    fine_tune_epochs = 10
    batch_size       = 32
    device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transforms
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # DataLoaders
    root = '/opt/img/effdl-cifar10/'
    train_loader = DataLoader(
        CIFAR10(root, train=True,  download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True,  num_workers=4
    )
    test_loader  = DataLoader(
        CIFAR10(root, train=False, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=4
    )

    results = []
    for prune_amount in prune_amounts:
        # 1) Charger le modèle
        model = ResNet18().to(device)
        model.load_state_dict(
            torch.load("best_baseline_resnet18.pth", map_location=device)
        )

        # 2) Forward/Backward sur un batch pour remplir .grad
        model.train()
        criterion = nn.CrossEntropyLoss()
        opt_prune = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt_prune.zero_grad()

        imgs, lbls = next(iter(train_loader))
        imgs, lbls = imgs.to(device), lbls.to(device)
        loss = criterion(model(imgs), lbls)
        loss.backward()

        # 3) Pruning structuré basé sur L2 des gradients
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # gradient shape [out_c, in_c, k, k] → norme L2 par filtre
                grads = m.weight.grad.view(m.out_channels, -1)
                grad_norms = grads.norm(p=2, dim=1)
                thresh = torch.quantile(grad_norms, prune_amount)
                # masque 1D → 4D
                keep = (grad_norms > thresh).to(torch.int)
                mask = keep[:, None, None, None].expand_as(m.weight)
                prune.custom_from_mask(m, 'weight', mask)

            elif isinstance(m, nn.Linear):
                # gradient shape [out_f, in_f] → norme L2 par neurone (ligne)
                grads = m.weight.grad  # [out_f, in_f]
                grad_norms = grads.norm(p=2, dim=1)
                thresh = torch.quantile(grad_norms, prune_amount)
                keep = (grad_norms > thresh).to(torch.int)
                mask = keep[:, None].expand_as(m.weight)
                prune.custom_from_mask(m, 'weight', mask)

        # 4) Fine-tuning
        opt_ft = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(fine_tune_epochs):
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                opt_ft.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                opt_ft.step()

        # 5) Évaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                _, preds = out.max(1)
                correct += preds.eq(lbls).sum().item()
                total += lbls.size(0)
        accuracy = 100. * correct / total

        # 6) Calcul du score (ps=prune_amount, pu=0)
        

        score = compute_model_score(model, ps=prune_amount, pu=0.0, qw=32, qa=32)

        results.append({
            'prune_amount': prune_amount,
            'accuracy':     accuracy,
            'score':        score
        })
        print(f"Ratio {prune_amount:.2f} → acc: {accuracy:.2f}%, score: {score:.4f}")

    # 7) Sauvegarde des résultats
    pd.DataFrame(results).to_csv('results.csv', index=False)
    print("Résultats enregistrés dans results.csv")

if __name__ == '__main__':
    main()
