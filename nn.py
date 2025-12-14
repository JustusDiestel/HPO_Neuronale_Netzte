import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from data import X_train_df, y_train, X_test_df, y_test, K

torch.manual_seed(0)

X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
X_test  = torch.tensor(X_test_df.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test  = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = nn.Sequential(
    nn.Linear(561, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, K)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)








num_epochs = 20

for epoch in range(num_epochs):
    # ===== Training =====
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)

        _, preds = torch.max(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss /= total
    train_acc = correct / total

    # ===== Evaluation =====
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            loss = criterion(logits, y)

            test_loss += loss.item() * x.size(0)

            _, preds = torch.max(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    test_loss /= total
    test_acc = correct / total

    print(
        f"Epoch {epoch+1:02d}/{num_epochs} | "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
    )
