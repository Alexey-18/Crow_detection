import cv2
from ultralytics import YOLO
import numpy as np
import torch
import supervision as sv

INPUT_FILE = 'crowd.mp4'
OUTPUT_FILE = 'output.mp4'
CONF_THRESHOLD = 0.35

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_path='yolo11m.pt'):
    model = YOLO(model_path)
    model.to(device)
    return model

def preprocess_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def detect_objects(frame, model, conf_threshold=0.35, base_min_size=50):
    height, width = frame.shape[:2]
    
    results = model(frame, verbose=False, device=device)
    detections = sv.Detections.from_ultralytics(results[0])
    
    mask = detections.class_id == 0
    detections = detections[mask]
    detections = detections[detections.confidence >= conf_threshold]
    
    min_width = max(base_min_size, int(width * 0.04))
    min_height = max(base_min_size, int(height * 0.06))
    
    widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    size_mask = (widths >= min_width) & (heights >= min_height)
    detections = detections[size_mask]
    
    if len(detections) > 0:
        widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
        heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
        
        small_mask = (widths < min_width * 1.5) | (heights < min_height * 1.5)
        detections.confidence[small_mask] = detections.confidence[small_mask] * 0.9
        
        right_zone_mask = detections.xyxy[:, 0] > width * 0.65
        detections = detections[~right_zone_mask]
    
    for xyxy, conf in zip(detections.xyxy, detections.confidence):
        x1, y1, x2, y2 = map(int, xyxy)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        
        label = f'{conf:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def process_video(input_path, output_path, conf_threshold=0.35, min_box_size=50):
    model = load_model()
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f'Ошибка открытия: {input_path}')
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        enhanced_frame = preprocess_frame(frame)
        annotated_frame = detect_objects(
            enhanced_frame, model, conf_threshold, min_box_size
        )
        
        out.write(annotated_frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f'Обработано {frame_count} кадров')
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f'Выходное видео сохранено: {output_path}')

if __name__ == '__main__':
    process_video(INPUT_FILE, OUTPUT_FILE, CONF_THRESHOLD)
