
import torch
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.device_count())  # Should return the number of CUDA devices
print(torch.cuda.get_device_name(0)) 