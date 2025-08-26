from ultralytics import YOLO

# ===== CONFIGURAÇÕES =====
dataset_yaml = "data.yaml"       # seu arquivo YAML
base_model = "yolo11n.pt"        # modelo leve e rápido
epochs = 200                     # suficiente para dataset pequeno
img_size = 640
batch_size = 8                   # seguro para 4GB VRAM da GTX 1650
device = "cuda"                  # usa GPU
train_name = "treino_v1_otimizado"
augment = True                    # ativa data augmentation para melhorar generalização
workers = 0                       # evita problemas no Windows
# ==========================

# Carregar modelo base
model = YOLO(base_model)
print("Iniciando treino otimizado...")

# Treinar modelo
train_results = model.train(
    data=dataset_yaml,
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    device=device,
    name=train_name,
    augment=augment,
    workers=workers
)

print("Treino finalizado!")
print(f"Pesos treinados salvos em: runs/train/{train_name}/weights/best.pt")
