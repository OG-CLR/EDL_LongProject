import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from resnet import ResNet18
import pandas as pd
from thop import profile


def compute_score(ps, pu, qw, qa, w, f):
    # Référence: w_ref=5.6e6/11173962, f_ref=2.8e8/278940160.0
    ref_w, ref_f = 11173962, 278940160.0
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

def main():
    # Hyperparamètres
    prune_amounts = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.96, 0.99]
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


       # === Bloc 1 : forward/backward pour le pruning global par gradient ===
        criterion = nn.CrossEntropyLoss()
        opt_prune = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt_prune.zero_grad()
        model.train()

        # on prend un batch pour remplir .grad
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        loss.backward()

        # on collecte tous les |grad| dans un seul vecteur
        all_grads = []
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight.grad is not None:
                all_grads.append(m.weight.grad.abs().view(-1))
        all_grads = torch.cat(all_grads)

        # seuil global au quantile prune_amount
        thresh = torch.quantile(all_grads, prune_amount)

        # application du même masque à chaque couche
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight.grad is not None:
                mask = (m.weight.grad.abs() > thresh).to(torch.int)
                prune.custom_from_mask(m, 'weight', mask)


        # === Bloc 2 : fine-tuning après pruning ===
        opt_ft = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        model.train()

        for epoch in range(fine_tune_epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                opt_ft.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()  
                opt_ft.step()


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
        w, f = compute_model_size(model, input_size=(1,3,32,32), device=device)

        score = compute_score(ps=0.0, pu=prune_amount, qw=32, qa=32, w=w, f=f)

        results.append({'prune_amount': prune_amount, 'accuracy': accuracy, 'score': score})
        print(f"Ratio {prune_amount:.2f} → acc: {accuracy:.2f}%, score: {score:.4f}")

    # Sauvegarde des résultats
    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)
    print("Résultats enregistrés dans results.csv")

if __name__ == '__main__':
    main()
