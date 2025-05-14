import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import pandas as pd
from resnet import ResNet18
from thop import profile

# === 🔧 Hyperparamètres ===
OPTIMIZER_NAME    = "SGD"
LEARNING_RATE     = 0.01
SCHEDULER         = True
DATA_AUGMENTATION = True
NESTEROV          = True
MOMENTUM          = 0.9
WEIGHT_DECAY      = 5e-4
BATCH_SIZE        = 32
EPOCHS            = 100

# Pruning & Quantization flags (for score calculation)
PS = 0.0    # structured pruning fraction
PU = 0.0    # unstructured pruning fraction
QW = 32     # weight bit-width
QA = 32     # activation bit-width

# === 📂 Chemins de sauvegarde ===
MODEL_DIR            = "saved_models"
BASELINE_MODEL_PATH  = os.path.join(MODEL_DIR, "best_baseline_resnet18.pth")
RESULTS_CSV          = "baseline_results.csv"
os.makedirs(MODEL_DIR, exist_ok=True)

# Prepare CSV if missing
if not os.path.exists(RESULTS_CSV):
    pd.DataFrame(columns=[
        'optimizer', 'learning_rate', 'scheduler', 'data_augmentation',
        'nesterov', 'momentum', 'weight_decay',
        'final_accuracy', 'duration', 'final_score'
    ]).to_csv(RESULTS_CSV, index=False)

# === 🔄 Data transforms ===
normalize = transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
]) if DATA_AUGMENTATION else transforms.Compose([
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# === 📦 Datasets & loaders ===
rootdir   = '/opt/img/effdl-cifar10/'
train_ds  = CIFAR10(rootdir, train=True,  download=True, transform=transform_train)
test_ds   = CIFAR10(rootdir, train=False, download=True, transform=transform_test)
trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
testloader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# === 🔍 Model init ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18().to(device)

# === ⚙️ Optimizer & scheduler ===
if OPTIMIZER_NAME == "SGD":
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=NESTEROV
    )
else:
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) if SCHEDULER else None

criterion = nn.CrossEntropyLoss()

# === 📌 Utility functions ===
def compute_model_size(model):
    w = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    f = 2.8e8  # fixed for ResNet-18
    return w, f

def compute_score(ps, pu, qw, qa, w, f):
    ref_w, ref_f = 5.6e6, 2.8e8
    mem  = (1 - (ps + pu)) * (qw/32) * (w/ref_w)
    comp = (1 - ps)       * (max(qw,qa)/32) * (f/ref_f)
    return mem + comp

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total   += targets.size(0)
    return 100. * correct / total

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

# === 🚀 Training loop ===
print("🚀 Starting training...")
w = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # nombre de MACs
f = compute_macs(model, device=device)

print(w, f)


global_best = 0.0
start_time  = time.time()

# to log per-epoch stats
epoch_stats = []

for epoch in range(1, EPOCHS+1):
    print(f"\n🌟 Epoch {epoch}/{EPOCHS}")
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds  = outputs.max(1)
        correct   += preds.eq(targets).sum().item()
        total     += targets.size(0)

    train_loss = running_loss / total
    train_acc  = 100. * correct / total
    print(f"🔄 Train  - loss: {train_loss:.4f}, acc: {train_acc:.2f}%")

    if scheduler:
        scheduler.step()

    # evaluation
    test_acc = evaluate(model, testloader, device)
    print(f"✅ Test   - acc: {test_acc:.2f}%")

    # compute epoch score
    w, f   = compute_model_size(model)
    escore = compute_score(PS, PU, QW, QA, w, f)
    print(f"🔢 Score  - epoch {epoch}: {escore:.4f}")

    epoch_stats.append({
        'epoch':    epoch,
        'train_acc':train_acc,
        'test_acc': test_acc,
        'score':    escore
    })

    # save best
    if test_acc > global_best:
        global_best = test_acc
        torch.save(model.state_dict(), BASELINE_MODEL_PATH)
        print(f"💾 Saved best model ({global_best:.2f}%)")

# total duration
duration = time.time() - start_time

# final score (should match last epoch escore)
final_score = compute_score(PS, PU, QW, QA, *compute_model_size(model))

# === 💾 Save final summary ===
result = {
    'optimizer':         OPTIMIZER_NAME,
    'learning_rate':     LEARNING_RATE,
    'scheduler':         SCHEDULER,
    'data_augmentation': DATA_AUGMENTATION,
    'nesterov':          NESTEROV,
    'momentum':          MOMENTUM,
    'weight_decay':      WEIGHT_DECAY,
    'final_accuracy':    global_best,
    'duration':          duration,
    'final_score':       final_score
}
pd.DataFrame([result]).to_csv(RESULTS_CSV, mode='a', header=False, index=False)
print(f"\n✅ Training complete  Best Acc: {global_best:.2f}%, Score: {final_score:.4f}")
print(f"📊 Epoch-by-epoch stats saved to 'epoch_results.csv'")

# === 💾 Save per-epoch stats ===
pd.DataFrame(epoch_stats).to_csv('epoch_results.csv', index=False)
