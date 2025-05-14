import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from binary_connect_class import BC
from resnet import ResNet18

def compute_score(ps, pu, qw, qa, w, f):
    ref_w, ref_f = 5.6e6, 2.8e8
    mem  = (1 - (ps + pu)) * (qw/32) * (w/ref_w)
    comp = (1 - ps) * (max(qw,qa)/32) * (f/ref_f)
    return mem + comp



def train_epoch(model, bc, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        bc.binarization()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        bc.restore()
        optimizer.step()
        bc.clip()
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, 100. * correct / total


def validate(model, bc, loader, criterion, device):
    model.eval()
    bc.binarization()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
    bc.restore()
    return val_loss / total, 100. * correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyperparams
    num_epochs = 50
    batch_size = 128
    lr = 1e-3

    # Data
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    val_ds   = datasets.CIFAR10('./data', train=False,download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # Model + BinaryConnect

    
    model = ResNet18().to(device)
    model.load_state_dict(torch.load("best_baseline_resnet18.pth", map_location=device))

    w     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f_mac = 2.8e8
   
    bc = BC(model)

     # ─── Monkey-patch BC.binarization to call save_params() ───────────

    def fixed_binarization():
        # save the current full-precision weights
        bc.save_params()
        # then binarize exactly as intended
        for i in range(bc.num_of_params):
            bw = bc.saved_params[i].sign()
            bc.target_modules[i].data.copy_(bw)

    # override the broken method
    bc.binarization = fixed_binarization

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    best_acc = 0.0
    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, bc, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, bc, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start_time


        score = compute_score(ps=0.0, pu=0.0, qw=1, qa=32, w=w, f=f_mac)

        print(f"Epoch {epoch:02d}/{num_epochs}  "
              f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}%  "
              f"Val: loss={val_loss:.4f}, acc={val_acc:.2f}%  "
              f"Score: {score:.4f}  ({elapsed:.1f}s)")


        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_binaryconnect.pth')

    print(f"Best Val Acc: {best_acc:.2f}% (model saved)")

if __name__ == '__main__':
    main()
