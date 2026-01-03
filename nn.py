import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import data
from data import X_train_df, X_val_df, X_test_df, K


def do_nn_training(
    individual,
    return_predictions=False,
    train_final_model=False,
):

    # === Hyperparameter ===
    num_layers    = individual[0]
    base_units    = individual[1]
    width_pattern = individual[2]
    activation    = individual[3]
    learning_rate = individual[4]
    optimizer_name = individual[5]
    batch_size    = individual[6]
    dropout_rate  = individual[7]
    weight_decay  = individual[8]
   # scaler_type   = individual[9]  Bei dem verwendeten Datensatz nicht nötig da er bereits skaliert ist
    scaler_type = "none"
    num_epochs    = int(individual[10])

    #Datenvorbereitung
    X_train = X_train_df.values
    X_val   = X_val_df.values
    X_test  = X_test_df.values

    if scaler_type == "standard":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)
    elif scaler_type == "none":
        pass
    else:
        raise ValueError("Unknown scaler")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(data.y_train.values, dtype=torch.long)
    y_val   = torch.tensor(data.y_val.values, dtype=torch.long)
    y_test  = torch.tensor(data.y_test.values, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    #Modellaufbau
    layers = []
    prev_units = X_train.shape[1]

    for i in range(num_layers):
        if width_pattern == "constant":
            units = base_units
        elif width_pattern == "increasing":
            units = base_units * (i + 1)
        elif width_pattern == "decreasing":
            units = base_units * (num_layers - i)
        else:
            raise ValueError("Invalid width pattern")

        layers.append(nn.Linear(prev_units, units))

        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "elu":
            layers.append(nn.ELU())
        else:
            raise ValueError("Invalid activation")

        layers.append(nn.Dropout(dropout_rate))
        prev_units = units


    layers.append(nn.Linear(prev_units, K))

    model = nn.Sequential(*layers)

    # optimierer und loss funktion
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer")

    for epoch in range(num_epochs):
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
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        train_loss /= total
        train_acc = correct / total

        # ---- Validation ----
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                correct += (model(x).argmax(dim=1) == y).sum().item()
                total += y.size(0)

        val_acc = correct / total

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    if return_predictions:
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test, dtype=torch.float32))
            y_pred = logits.argmax(dim=1)
        return y_pred.numpy()

    return val_acc