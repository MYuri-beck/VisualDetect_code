from ultralytics import YOLO

# ===== CONFIGURAÇÕES =====
checkpoint = "runs/detect/treino_v1_otimizado/weights/best.pt"  # pesos base
dataset_yaml = "data.yaml"          # seu arquivo YAML
epochs = 250                      # novas épocas para continuar o treino
img_size = 640
batch_size = 8
device = "cuda"                      # GPU ou "cpu" se não tiver
train_name = "treino_v2.1" # nova pasta para não sobrescrever
augment = True
workers = 0
# ==========================

# Carregar modelo a partir do checkpoint
model = YOLO(checkpoint)
print("Continuando o treino a partir do checkpoint...")

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

print("Treino continuado finalizado!")
print(f"Novos pesos salvos em: runs/train/{train_name}/weights/best.pt")
