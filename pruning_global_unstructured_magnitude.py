import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from resnet import ResNet18
import pandas as pd


def compute_score(ps, pu, qw, qa, w, f):
    # Référence: w_ref=5.6e6, f_ref=2.8e8
    ref_w, ref_f = 5.6e6, 2.8e8
    mem = (1 - (ps + pu)) * (qw / 32) * (w / ref_w)
    comp = (1 - ps) * (max(qw, qa) / 32) * (f / ref_f)
    return mem + comp


def main():
    # Hyperparamètres
    prune_amounts = [ 0.97, 0.98, 0.99]
    fine_tune_epochs = 10
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))

    # Toujours data augmentation sur le train
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    ])

    # Pas d'augmentation sur le test
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
    ])

    # Chargement
    rootdir = '/opt/img/effdl-cifar10/'
    trainset = CIFAR10(rootdir, train=True, download=True, transform=transform_train)
    testset  = CIFAR10(rootdir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=4)

    results = []
    for prune_amount in prune_amounts:
        # Chargement du modèle pré-entraîné
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResNet18().to(device)
        model.load_state_dict(torch.load("best_baseline_resnet18.pth", map_location=device))


        # Pruning unstructured L1 GLOBAL
        parameters_to_prune = [
            (m, 'weight')
            for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))
        ]
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=prune_amount,
        )


        # Fine-tuning
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for epoch in range(fine_tune_epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

        # Évaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        accuracy = 100. * correct / total

        # Calcul du score
        w = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f = f = 2.8e8
        score = compute_score(ps=0.0, pu=prune_amount, qw=32, qa=32, w=w, f=f)

        results.append({'prune_amount': prune_amount, 'accuracy': accuracy, 'score': score})
        print(f"Ratio {prune_amount:.2f} → acc: {accuracy:.2f}%, score: {score:.4f}")

    # Sauvegarde des résultats
    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)
    print("Résultats enregistrés dans results.csv")

if __name__ == '__main__':
    main()
