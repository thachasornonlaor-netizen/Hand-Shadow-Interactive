import cv2
import socket
from collections import deque, Counter
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r"C:\Users\thach\Downloads\train6\train6\weights\best.pt")

# Set up UDP socket
udp_ip = "127.0.0.1"
udp_port = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Open webcam stream
cap = cv2.VideoCapture(0)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.75

# Class label to UDP format (without id)
class_to_udp = {
    "bird": "0,1,0,0,0",
    "moose": "1,0,1,0,0",
    "panther": "2,0,0,1,0",
    "snail": "3,0,0,0,1"
}

# Keep last 20 frames of detected (label, id) pairs
frame_buffer = deque(maxlen=20)
last_sent = {}  # Store sent (label, id) pairs to avoid repeated sending

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=CONFIDENCE_THRESHOLD, tracker="bytetrack.yaml")
    
    current_frame_pairs = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            label = model.names[cls_id]
            track_id = int(box.id[0]) if box.id is not None else -1
            current_frame_pairs.append((label, track_id))

            # Draw box and label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} conf:{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Update buffer
    frame_buffer.append(current_frame_pairs)

    # Count (label, id) pair occurrences in last 20 frames
    pair_counter = Counter(pair for frame in frame_buffer for pair in frame)

    for (label, track_id), count in pair_counter.items():
        if label in class_to_udp and count >= 16:  # 80% of 20 frames
            # Check if this object has already been sent
            if (label, track_id) not in last_sent:
                udp_message = f"{track_id},{class_to_udp[label]}"
                sock.sendto(udp_message.encode(), (udp_ip, udp_port))
                print(f"[UDP SENT] {udp_message}")
                # Mark the object as sent by storing the track_id and label
                last_sent[(label, track_id)] = True
            break
    else:
        last_sent.clear()  # Reset if nothing met threshold

    cv2.imshow("YOLOv8 Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
sock.close()
