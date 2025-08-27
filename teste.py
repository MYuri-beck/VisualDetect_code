import cv2
from ultralytics import YOLO

model_path = "runs/detect/treino_v1_continuacao/weights/best.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: não foi possível acessar a webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predição mais rápida
    for result in model(frame, stream=True):
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("YOLO Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
