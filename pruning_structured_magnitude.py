import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from resnet import ResNet18
import pandas as pd


def compute_score(ps, pu, qw, qa, w, f):
    # Référence: w_ref=5.6e6, f_ref=2.8e8
    ref_w, ref_f = 5.6e6, 2.8e8
    mem  = (1 - (ps + pu)) * (qw / 32) * (w / ref_w)
    comp = (1 - ps) * (max(qw, qa) / 32) * (f / ref_f)
    return mem + comp


def main():
    # Hyperparamètres
    prune_amounts    = [0..3]
    fine_tune_epochs = 5
    batch_size       = 32
    device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    trainset     = CIFAR10('/opt/img/effdl-cifar10/', train=True,  download=True, transform=transform_train)
    testset      = CIFAR10('/opt/img/effdl-cifar10/', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=4)

    results = []
    for prune_amount in prune_amounts:
        # — Charger le modèle
        model = ResNet18().to(device)
        model.load_state_dict(torch.load("best_baseline_resnet18.pth", map_location=device))

        # — Pruning structuré L2
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # supprime prune_amount fraction des filtres (dim=0) d'après norm L2
                prune.ln_structured(m, name='weight',
                                   amount=prune_amount, n=2, dim=0)
            elif isinstance(m, nn.Linear):
                # supprime prune_amount fraction des neurones (dim=1) d'après norm L2
                prune.ln_structured(m, name='weight',
                                   amount=prune_amount, n=2, dim=1)

        # — Fine-tuning
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(fine_tune_epochs):
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()

        # — Évaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                _, preds = out.max(1)
                correct += preds.eq(labels).sum().item()
                total   += labels.size(0)
        accuracy = 100. * correct / total

        # — Calcul du score (ici ps=prune_amount, pu=0)
        w     = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f_mac = 2.8e8
        score = compute_score(ps=prune_amount, pu=0.0, qw=32, qa=32, w=w, f=f_mac)

        results.append({
            'prune_amount': prune_amount,
            'accuracy':     accuracy,
            'score':        score
        })
        print(f"Ratio {prune_amount:.2f} → acc: {accuracy:.2f}%, score: {score:.4f}")

    # — Sauvegarde
    pd.DataFrame(results).to_csv('results.csv', index=False)
    print("Résultats enregistrés dans results.csv")


if __name__ == '__main__':
    main()
