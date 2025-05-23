import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from resnet_2 import ResNet18
import pandas as pd
from thop import profile
from ptflops import get_model_complexity_info
import torch.optim as optim

LEARNING_RATE     = 0.01
SCHEDULER         = True
NESTEROV          = True
MOMENTUM          = 0.9
WEIGHT_DECAY      = 5e-4
BATCH_SIZE        = 128
EPOCHS            = 20


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
    prune_amounts = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99]
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

        #Quantization
        model.half()

        # Garder BatchNorm en float32 pour la stabilité
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.float()


        # Pruning unstructured L1 local
        for module in model.modules():   
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=prune_amount)

        # Fine-tuning
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_tune_epochs) if SCHEDULER else None
        optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=NESTEROV
    )
        model.train()
        for epoch in range(fine_tune_epochs):
            for images, labels in train_loader:
                images, labels = images.to(device).half(), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
            if scheduler:
                scheduler.step()

        # Évaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device).half(), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        accuracy = 100. * correct / total

        # Calcul du score

        score = compute_model_score(model, ps=0.0, pu=prune_amount, qw=16, qa=16)

        results.append({'prune_amount': prune_amount, 'accuracy': accuracy, 'score': score})
        print(f"Ratio {prune_amount:.2f} → acc: {accuracy:.2f}%, score: {score:.4f}")

    # Sauvegarde des résultats
    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)
    print("Résultats enregistrés dans results.csv")

if __name__ == '__main__':
    main()