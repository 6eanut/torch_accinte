# torch==2.9.1
import torch

if not torch.accinte.is_available():
    print("accinte backend is not available in this build.")
    exit()

print("accinte backend is available!")

device = torch.device("accinte")

x = torch.tensor([[-1., -2.], [-3., -4.]], device=device)
y = torch.ops.accinte.custom_abs(x)
print("Result y:\n", y)
print(f"Device of y: {y.device}")

