import cv2
import mediapipe as mp
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback
)

def is_peace_sign(hand_landmarks):

    lm = hand_landmarks

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y

    return index_up and middle_up and ring_down and pinky_down

DISPLAY_WIDTH = 520
DISPLAY_HEIGHT = 405
cap = cv2.VideoCapture(0)

native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolusi native webcam: {native_width}x{native_height}")
print(f"R: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")

WINDOW_NAME = 'Peace web Sign v4'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

print("=" * 50)
print(" hello mf ")
print("=" * 50)

frame_timestamp = 0
printed_actual_size = False

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ada yang salah pas buka webcam!")
            break

        frame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        if not printed_actual_size:
            h, w = frame.shape[:2]
            print(f"Ukuran frame setelah resize: {w}x{h}")
            printed_actual_size = True
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_timestamp += 33
        landmarker.detect_async(mp_image, frame_timestamp)

        peace_detected = False
        if latest_result and latest_result.hand_landmarks:
            for hand_lm in latest_result.hand_landmarks:
                if is_peace_sign(hand_lm):
                    peace_detected = True
                    break

        if peace_detected:
            frame = cv2.GaussianBlur(frame, (35, 35), 10)

            cv2.putText(
                frame,
                "Just Friend Sih",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),   # Warna hijau
                2,
                cv2.LINE_AA
            )

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
