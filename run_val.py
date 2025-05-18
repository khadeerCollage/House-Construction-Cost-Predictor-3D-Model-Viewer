from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load a pretrained model
model.train(data='yolov5/data.yaml', epochs=50, imgsz=640)
