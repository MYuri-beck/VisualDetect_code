import torch

print("Versão do PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Nome da GPU:", torch.cuda.get_device_name(0))
    print("Número de GPUs disponíveis:", torch.cuda.device_count())
    print("Versão CUDA compilada no PyTorch:", torch.version.cuda)
