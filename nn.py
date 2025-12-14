import torch
import data


train_tensor = torch.as_tensor(data.get_train_data())
test_tensor = torch.as_tensor(data.get_test_data())
train_result_tensor = torch.as_tensor(data.get_train_result())
test_result_tensor = torch.as_tensor(data.get_test_result())


