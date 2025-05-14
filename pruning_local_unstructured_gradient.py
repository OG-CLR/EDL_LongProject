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
    prune_amounts    = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99]
    fine_tune_epochs = 10
    batch_size       = 32
    device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    # Transforms
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

    # Datasets & Loaders
    rootdir    = '/opt/img/effdl-cifar10/'
    trainset   = CIFAR10(rootdir, train=True,  download=True, transform=transform_train)
    testset    = CIFAR10(rootdir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=4)

    results = []

    for prune_amount in prune_amounts:
        # 1) Chargement du modèle
        model = ResNet18().to(device)
        model.load_state_dict(torch.load("best_baseline_resnet18.pth", map_location=device))

        # === Bloc 1 : backward pour remplir .grad avant pruning ===
        model.train()
        criterion   = nn.CrossEntropyLoss()
        opt_prune   = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt_prune.zero_grad()

        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        loss.backward()

        # === Bloc 2 : pruning local basé sur L1 des gradients ===
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight.grad is not None:
                grad_abs = m.weight.grad.abs().view(-1)
                thresh   = torch.quantile(grad_abs, prune_amount)
                mask     = (m.weight.grad.abs() > thresh).to(torch.int)
                prune.custom_from_mask(m, 'weight', mask)

        # === Bloc 3 : fine-tuning après pruning ===
        opt_ft   = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(fine_tune_epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                opt_ft.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                opt_ft.step()

        # === Évaluation ===
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total   += labels.size(0)
        accuracy = 100. * correct / total

        # === Calcul du score ===
        

        score = compute_model_score(model, ps=0.0, pu=prune_amount, qw=32, qa=32)

        results.append({
            'prune_amount': prune_amount,
            'accuracy':     accuracy,
            'score':        score
        })
        print(f"Ratio {prune_amount:.2f} → acc: {accuracy:.2f}%, score: {score:.4f}")

    # Sauvegarde
    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)
    print("Résultats enregistrés dans results.csv")


if __name__ == '__main__':
    main()
