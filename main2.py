'''Training CIFAR10 through four stages: normal, data aug + scheduler, + structured pruning, + unstructured pruning'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import pandas as pd

from binary_connect import BC
from models.resnet import ResNet18
import torch.nn.utils.prune as prune
from utils import progress_bar

# Global settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 100
csv_path = "train_results_all.csv"

# Score computation
def cifar10_score(ps, pu, qw, qa, w, f):
    ref_w, ref_f = 5.6e6, 2.8e8
    mem = (1 - (ps + pu)) * (qw / 32) * (w / ref_w)
    comp = (1 - ps) * (max(qw, qa) / 32) * (f / ref_f)
    return mem + comp

def estimate_wf(ps, pu):
    base_w = 5.6e6
    base_f = 2.8e8
    w = base_w * (1 - ps) * (1 - 0.5 * pu)
    f = base_f * (1 - ps)
    return int(w), int(f)

# Dataloader builder
def build_dataloaders(data_aug=False):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    np.random.seed(2147483647)
    indices = np.random.permutation(len(trainset))[:15000]
    train_subset = torch.utils.data.Subset(trainset, indices)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader

# Model builder
def build_model():
    model = ResNet18().to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    return model

# Optimizer and scheduler builder
def build_optimizer_scheduler(model, use_scheduler=False):
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None
    return optimizer, scheduler

# Training and test functions
def train_one_epoch(model, trainloader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), f'Loss: {train_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.2f}% ({correct}/{total})')

def test_model(model, testloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

# Pruning helpers
def apply_structured_pruning(model, amount):
    for layer in [model.module.layer1, model.module.layer2, model.module.layer3, model.module.layer4]:
        for block in layer:
            prune.ln_structured(block.conv1, name='weight', amount=amount, n=2, dim=0)
            prune.ln_structured(block.conv2, name='weight', amount=amount, n=2, dim=0)
            if hasattr(block, 'downsample') and block.downsample is not None:
                for m in block.downsample:
                    if isinstance(m, nn.Conv2d):
                        prune.ln_structured(m, name='weight', amount=amount, n=2, dim=0)

def apply_unstructured_pruning(model, amount):
    parameters_to_prune = []
    for layer in [model.module.layer1, model.module.layer2, model.module.layer3, model.module.layer4]:
        for block in layer:
            parameters_to_prune.append((block.conv1, 'weight'))
            parameters_to_prune.append((block.conv2, 'weight'))
            if hasattr(block, 'downsample') and block.downsample is not None:
                for m in block.downsample:
                    if isinstance(m, nn.Conv2d):
                        parameters_to_prune.append((m, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )

def run_pruning_with_finetuning(pruning_ratio=0.97, fine_tune_epochs=5):
    print(f"\n🧪 Structured experiment: Unstructured pruning à {int(pruning_ratio*100)}% + fine-tuning ({fine_tune_epochs} epochs)")
    trainloader, testloader = build_dataloaders(data_aug=True)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    
    # Step 1: Train ResNet18 for baseline
    optimizer, scheduler = build_optimizer_scheduler(model, use_scheduler=True)
    for epoch in range(num_epochs):
        train_one_epoch(model, trainloader, optimizer, criterion)
        if scheduler:
            scheduler.step()

    # Step 2: Apply unstructured pruning
    to_prune = [(m, 'weight') for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(to_prune, pruning_method=prune.L1Unstructured, amount=pruning_ratio)
    print(f"🔧 Pruning {int(pruning_ratio * 100)}% appliqué")

    # Step 3: Accuracy before fine-tuning
    acc_before = test_model(model, testloader, criterion)
    print(f"📉 Accuracy avant fine-tuning : {acc_before:.2f}%")

    # Step 4: Fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    for epoch in range(1, fine_tune_epochs + 1):
        train_one_epoch(model, trainloader, optimizer, criterion)
        sparsity = compute_sparsity_with_mask(model)
        print(f"🌿 Epoch {epoch} — Sparsité actuelle : {sparsity:.2f}%")

    # Step 5: Accuracy après fine-tuning
    acc_after = test_model(model, testloader, criterion)
    print(f"✅ Accuracy après fine-tuning : {acc_after:.2f}%")

    # Step 6: Score final
    ps, pu = 0.0, pruning_ratio  # ici pas de structured pruning
    w, f = estimate_wf(ps, pu)
    score = cifar10_score(ps, pu, 32, 32, w, f)

    return {
        'data_aug': True,
        'scheduler': True,
        'p_s': ps,
        'p_u': pu,
        'accuracy_before_ft': acc_before,
        'accuracy_after_ft': acc_after,
        'score': score,
        'sparsity': sparsity,
    }

def compute_sparsity_with_mask(model):
    total = 0
    zeros = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask
                zeros += torch.sum(mask == 0).item()
                total += mask.numel()
            else:
                param = module.weight
                zeros += torch.sum(param == 0).item()
                total += param.numel()
    return 100. * zeros / total

def run_float16_quantization_test():
    print("\n🧪 Test quantization float16 sur ResNet18")
    trainloader, testloader = build_dataloaders(data_aug=True)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    
    # 1. Entraînement classique en float32
    optimizer, scheduler = build_optimizer_scheduler(model, use_scheduler=True)
    for epoch in range(num_epochs):
        train_one_epoch(model, trainloader, optimizer, criterion)
        if scheduler:
            scheduler.step()

    # 2. Évaluation float32
    acc_fp32 = test_model(model, testloader, criterion)
    print(f"📊 Accuracy float32 : {acc_fp32:.2f}%")

    # 3. Conversion en float16
    model_fp16 = model.half()
    print("🔧 Conversion du modèle en float16")

    # 4. Évaluation float16
    model_fp16.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.half()  # très important
            outputs = model_fp16(inputs)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
    acc_fp16 = 100. * correct / total
    print(f"📊 Accuracy float16 : {acc_fp16:.2f}%")

    # 5. Score (aucun pruning donc ps = pu = 0)
    ps, pu = 0.0, 0.0
    w, f = estimate_wf(ps, pu)
    score = cifar10_score(ps, pu, 16, 16, w, f)  # float16 = quantized 16 bits

    return {
        'data_aug': True,
        'scheduler': True,
        'p_s': ps,
        'p_u': pu,
        'accuracy_fp32': acc_fp32,
        'accuracy_fp16': acc_fp16,
        'delta_accuracy': acc_fp32 - acc_fp16,
        'quantized': 'float16',
        'score': score,
        'w_est': w,
        'f_est': f
    }

def run_binaryconnect_training():
    print("\n🧪 Test BinaryConnect (BC) sur ResNet18")
    trainloader, testloader = build_dataloaders(data_aug=True)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bc = BC(model)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            bc.binarization()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            bc.restore()
            optimizer.step()
            bc.clip()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        acc = 100. * correct / total
        print(f"[Epoch {epoch}] BC Train Accuracy: {acc:.2f}%")

    bc.binarization()  # Evaluation must use binary weights
    final_acc = test_model(model, testloader, criterion)
    bc.restore()

    ps, pu = 0.0, 0.0
    w, f = estimate_wf(ps, pu)
    score = cifar10_score(ps, pu, qw=1, qa=1, w=w, f=f)

    print(f"✅ Accuracy après BinaryConnect : {final_acc:.2f}%")
    print(f"📈 Score BinaryConnect : {score:.4f}")

    return {
        'data_aug': True,
        'scheduler': False,
        'p_s': ps,
        'p_u': pu,
        'accuracy': final_acc,
        'quantized': 'binaryconnect',
        'score': score,
        'w_est': w,
        'f_est': f
    }



# Main training procedure for one config
def run_training(data_aug=False, use_scheduler=False, structured_p=0.0, unstructured_p=0.0):
    print(f"\n🚀 Training with settings: DataAug={data_aug}, Scheduler={use_scheduler}, P_s={structured_p}, P_u={unstructured_p}")
    trainloader, testloader = build_dataloaders(data_aug)
    model = build_model()
    optimizer, scheduler = build_optimizer_scheduler(model, use_scheduler)
    criterion = nn.CrossEntropyLoss()

    # Apply pruning if needed
    if structured_p > 0:
        apply_structured_pruning(model, amount=structured_p)
    if unstructured_p > 0:
        apply_unstructured_pruning(model, amount=unstructured_p)

    for epoch in range(num_epochs):
        train_one_epoch(model, trainloader, optimizer, criterion)
        if scheduler:
            scheduler.step()

    final_acc = test_model(model, testloader, criterion)
    w, f = estimate_wf(structured_p, unstructured_p)
    score = cifar10_score(structured_p, unstructured_p, 32, 32, w, f)

    print(f"✅ Final Test Accuracy: {final_acc:.2f}%")
    print(f"📈 Final Score: {score:.4f}")

    return {
        'data_aug': data_aug,
        'scheduler': use_scheduler,
        'p_s': structured_p,
        'p_u': unstructured_p,
        'accuracy': final_acc,
        'score': score,
    }


# Grille de taux de pruning à explorer
structured_pruning_vals = [0.0, 0.3, 0.4, 0.5]
unstructured_pruning_vals = [0.0, 0.2, 0.3]


# Valeurs de pruning non structuré pour fine-tuning
fine_tune_pruning_ratios = [0.7, 0.8, 0.9, 0.95, 0.97]

results = []

# Expériences classiques sans pruning
results.append(run_training(data_aug=False, use_scheduler=False))
results.append(run_training(data_aug=True, use_scheduler=True))

# Pruning combinatoire
for p_s in structured_pruning_vals:
    for p_u in unstructured_pruning_vals:
        if p_s == 0.0 and p_u == 0.0:
            continue  # déjà testée plus haut
        results.append(run_training(data_aug=True, use_scheduler=True, structured_p=p_s, unstructured_p=p_u))

# Pruning + fine-tuning pour plusieurs ratios
for ratio in fine_tune_pruning_ratios:
    results.append(run_pruning_with_finetuning(pruning_ratio=ratio, fine_tune_epochs=10))


# Quantization test
results.append(run_float16_quantization_test())


#BinaryConect test
results.append(run_binaryconnect_training())


# Save all results
df_results = pd.DataFrame(results)
df_results.to_csv(csv_path, index=False)
print(f"\n📝 All results saved to {csv_path}")
