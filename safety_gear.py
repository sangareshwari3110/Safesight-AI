# ============================================
# PPE DETECTION (VEST + HELMET + SAFETY SHOES)
# ============================================

from ultralytics import YOLO
import cv2

# ------------------ MODELS ------------------
person_model = YOLO("yolov8n.pt")       # person detection
vest_model = YOLO("vest.pt")
helmet_model = YOLO("helmet.pt")
shoes_model = YOLO("safety_shoes.pt")

# ------------------ IMAGE -------------------
IMAGE_PATH = "unsafe.png"   # change your image path
image = cv2.imread(IMAGE_PATH)

MAX_HEIGHT = 800   # adjust if needed
h, w = image.shape[:2]

if h > MAX_HEIGHT:
    scale = MAX_HEIGHT / h
    new_w = int(w * scale)
    image = cv2.resize(image, (new_w, MAX_HEIGHT))

# ------------------ DETECT PERSONS ------------------
person_results = person_model(image)[0]

person_boxes = []
for box in person_results.boxes:
    cls = int(box.cls[0])
    if person_model.names[cls] == "person":
        person_boxes.append(box.xyxy[0].cpu().numpy())

# ------------------ DETECT PPE ------------------
vest_boxes = vest_model(image)[0].boxes.xyxy.cpu().numpy() if vest_model(image)[0].boxes else []
helmet_boxes = helmet_model(image)[0].boxes.xyxy.cpu().numpy() if helmet_model(image)[0].boxes else []
shoes_boxes = shoes_model(image)[0].boxes.xyxy.cpu().numpy() if shoes_model(image)[0].boxes else []

# ------------------ IOU FUNCTION ------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# ------------------ PROCESS EACH PERSON ------------------
for idx, person in enumerate(person_boxes):
    vest = any(iou(person, v) > 0.1 for v in vest_boxes)
    helmet = any(iou(person, h) > 0.1 for h in helmet_boxes)
    shoes = any(iou(person, s) > 0.1 for s in shoes_boxes)

    # Print result
    print(f"Person {idx+1}: Vest={vest}, Helmet={helmet}, SafetyShoes={shoes}")

    # Draw bounding box
    x1, y1, x2, y2 = map(int, person)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    vest = False
    helmet = False
    shoes = False
    label = f"Vest:{vest} Helmet:{helmet} Shoes:{shoes}"
    cv2.putText(image, label, (x1-100, y1 - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

# ------------------ SHOW RESULT ------------------
cv2.imshow("PPE Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
