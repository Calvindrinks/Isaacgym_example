import torch
import torch.nn as nn


input_size = 3
batch_size = 5
eps = 1e-1


class CustomBatchNorm1d:
    def __init__(self, weight, bias, eps, momentum, input_size):
        self.count = 0
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.momentum = momentum

        self.training_mode = True

        self.running_mean = torch.zeros(input_size)
        self.running_var = torch.ones(input_size) # it has Brassel's correction
        self.brassel_correction = batch_size / (batch_size - 1)

    def __call__(self, input_tensor):
        if not self.training_mode:
            dispersion = self.running_var
            math_expect = self.running_mean
        else:
            dispersion = input_tensor.var(unbiased=False, dim=0)
            math_expect = input_tensor.mean(dim=0)

            self.running_mean = self.momentum * math_expect + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * dispersion * self.brassel_correction + (1 - self.momentum) * self.running_var
            self.count += 1

            print(f"loop: {self.count} -> mean: {self.running_mean}")

        num = (input_tensor - math_expect)
        den = torch.sqrt(dispersion + eps)
        normed_tensor = (num / den) * self.weight + self.bias

        return normed_tensor

    def eval(self):
        self.training_mode = False


batch_norm = nn.BatchNorm1d(input_size, eps=eps)
batch_norm.bias.data = torch.randn(input_size, dtype=torch.float)
batch_norm.weight.data = torch.randn(input_size, dtype=torch.float)
batch_norm.momentum = 0.1
print(batch_norm.momentum)

custom_batch_norm1d = CustomBatchNorm1d(batch_norm.weight.data,
                                        batch_norm.bias.data, eps, batch_norm.momentum, input_size)

all_correct = True

for i in range(8):
    torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
    norm_output = batch_norm(torch_input)
    custom_output = custom_batch_norm1d(torch_input)
    all_correct &= torch.allclose(norm_output, custom_output, atol=1e-04) \
                   and norm_output.shape == custom_output.shape

print('\nmean: batch_norm -- ', batch_norm.running_mean, '\ncustom_batch_norm1d -- ', custom_batch_norm1d.running_mean)
print('\nvar: batch_norm -- ', batch_norm.running_var, '\ncustom_batch_norm1d -- ', custom_batch_norm1d.running_var)

batch_norm.eval()
custom_batch_norm1d.eval()

for i in range(8):
    torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
    norm_output = batch_norm(torch_input)
    custom_output = custom_batch_norm1d(torch_input)
    all_correct &= torch.allclose(norm_output, custom_output, atol=1e-04) \
                   and norm_output.shape == custom_output.shape


print(all_correct)