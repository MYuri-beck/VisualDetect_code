import cv2
from ultralytics import YOLO

# Caminho do modelo treinado
model_path = "runs/detect/treino_v1_continuacao/weights/best.pt"
model = YOLO(model_path)

# Caminho da imagem para teste
image_path = "valid\images\RR28_png.rf.1a388c3925fe066a7f2621cbd3eea336.jpg"  # troque pelo caminho da sua imagem
frame = cv2.imread(image_path)

if frame is None:
    print("Erro: não foi possível carregar a imagem")
    exit()

# Corrige espelhamento (caso queira manter o flip)
frame = cv2.flip(frame, 1)

# Predição
results = model(frame)

# Desenha as caixas na imagem
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        label = model.names[cls]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# Exibe a imagem
cv2.imshow("YOLO - Imagem", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()