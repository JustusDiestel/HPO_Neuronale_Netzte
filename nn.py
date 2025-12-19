import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import data
from data import X_train_df, y_train, X_test_df, y_test, K


def build_optimizer(optimizer_name, model, learning_rate, weight_decay):
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def do_nn_training(individual):

    num_layers = individual[0]  # num_layers
    base_units = individual[1]  # base_units
    width_pattern = individual[2]  # width_pattern
    activation = individual[3]  # activation
    learning_rate = individual[4]  # learning_rate
    optimizer_name = individual[5]  # optimizer
    batch_size = individual[6]  # batch_size
    dropout_rate = individual[7]  # dropout_rate
    weight_decay = individual[8]  # l2_weight_decay


    torch.manual_seed(0)

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    X_test  = torch.tensor(X_test_df.values, dtype=torch.float32)
    y_train = torch.tensor(data.y_train.values, dtype=torch.long)
    y_test  = torch.tensor(data.y_test.values, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = nn.Sequential()

    for _ in range(num_layers):
        if width_pattern == 'constant':
            units = base_units
        elif width_pattern == 'increasing':
            units = base_units * (_ + 1)
        elif width_pattern == 'decreasing':
            units = base_units * (num_layers - _)

        model.add_module(f'linear_{_}', nn.Linear(X_train.shape[1] if _ == 0 else prev_units, units))

        if activation == 'relu':
            model.add_module(f'activation_{_}', nn.ReLU())
        elif activation == 'tanh':
            model.add_module(f'activation_{_}', nn.Tanh())
        elif activation == 'elu':
            model.add_module(f'activation_{_}', nn.ELU())

        model.add_module(f'dropout_{_}', nn.Dropout(dropout_rate))

        prev_units = units

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(optimizer_name, model, learning_rate, weight_decay)

    num_epochs = 20

    for epoch in range(num_epochs):
        # ===== Training =====
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        test_acc_last = 0.0

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
        test_acc_last = test_acc

    return test_acc_last



