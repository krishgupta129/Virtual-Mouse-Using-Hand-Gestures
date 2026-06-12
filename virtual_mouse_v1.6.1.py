import cv2
import mediapipe as mp
import pyautogui
import math
import time
import os

# ---------------- SETTINGS ----------------

SENSITIVITY = 2.8
SMOOTHING = 0.35

CLICK_THRESHOLD = 30
CLICK_DELAY = 0.5

DRAG_HOLD_TIME = 0.7

RIGHT_CLICK_THRESHOLD = 35
RIGHT_CLICK_DELAY = 0.7

SCROLL_THRESHOLD = 25
SCROLL_DEADZONE = 8
SCROLL_SPEED = 4

# ------------------------------------------

pyautogui.FAILSAFE = False

screenshots_dir = os.path.join(
    os.path.expanduser("~"),
    "Pictures",
    "Screenshots"
)

os.makedirs(screenshots_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

prev_palm_x = None
prev_palm_y = None

# Virtual cursor
cursor_x, cursor_y = pyautogui.position()

last_click = 0
last_right_click = 0
last_release_click_time = 0
fps = 0
prev_frame_time = time.time()

pinch_start_time = None

dragging = False
drag_started = False

scroll_mode = False
scroll_start_y = None
flash_state = False

last_screenshot_time = 0
screenshot_flash_until = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    current_frame_time = time.time()
    fps = int(1 / max(current_frame_time - prev_frame_time, 0.0001))
    prev_frame_time = current_frame_time

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)


    # Two-hand OPEN PALM screenshot gesture
    if (
        results.multi_hand_landmarks and
        len(results.multi_hand_landmarks) == 2
    ):
        palms_open = 0

        for hnd in results.multi_hand_landmarks:
            lm = hnd.landmark

            if (
                lm[8].y < lm[6].y and
                lm[12].y < lm[10].y and
                lm[16].y < lm[14].y and
                lm[20].y < lm[18].y
            ):
                palms_open += 1

        if (
            palms_open == 2 and
            time.time() - last_screenshot_time > 2
        ):
            filename = os.path.join(
                screenshots_dir,
                f"screenshot_{int(time.time())}.png"
            )

            pyautogui.screenshot(filename)

            print(f"Screenshot saved: {filename}")

            last_screenshot_time = time.time()
            screenshot_flash_until = time.time() + 0.2

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = hand.landmark

        # Palm center
        palm = landmarks[9]

        palm_x = int(palm.x * w)
        palm_y = int(palm.y * h)

        # Fingers
        thumb = landmarks[4]
        index = landmarks[8]
        middle = landmarks[12]

        thumb_x = int(thumb.x * w)
        thumb_y = int(thumb.y * h)

        index_x = int(index.x * w)
        index_y = int(index.y * h)

        middle_x = int(middle.x * w)
        middle_y = int(middle.y * h)

        # ----------------------------
        # DISTANCES FIRST
        # ----------------------------

        # ----------------------------
        # DISTANCES
        # ----------------------------

        left_distance = math.hypot(
            thumb_x - index_x,
            thumb_y - index_y
        )

        right_distance = math.hypot(
            thumb_x - middle_x,
            thumb_y - middle_y
        )

        current_time = time.time()

        index_middle_distance = math.hypot(
            index_x - middle_x,
            index_y - middle_y
        )

        thumb_middle_distance = math.hypot(
            thumb_x - middle_x,
            thumb_y - middle_y
        )


        scroll_gesture = (
            index_middle_distance < SCROLL_THRESHOLD
            and left_distance > CLICK_THRESHOLD
            and thumb_middle_distance > RIGHT_CLICK_THRESHOLD
        )

        if scroll_gesture:

            if not scroll_mode:

                scroll_mode = True
                scroll_start_y = palm_y

            delta_y = scroll_start_y - palm_y

            if abs(delta_y) > SCROLL_DEADZONE:

                pyautogui.scroll(
                    int(delta_y * SCROLL_SPEED)
                )

                scroll_start_y = palm_y

        else:

            scroll_mode = False
            scroll_start_y = None

            if prev_palm_x is None:

                prev_palm_x = palm_x
                prev_palm_y = palm_y

            else:

                dx = palm_x - prev_palm_x
                dy = palm_y - prev_palm_y

                target_x = cursor_x + dx * SENSITIVITY
                target_y = cursor_y + dy * SENSITIVITY

                target_x = max(
                    0,
                    min(screen_w - 1, target_x)
                )

                target_y = max(
                    0,
                    min(screen_h - 1, target_y)
                )

                cursor_x += (target_x - cursor_x) * SMOOTHING
                cursor_y += (target_y - cursor_y) * SMOOTHING

                pyautogui.moveTo(cursor_x, cursor_y)

                prev_palm_x = palm_x
                prev_palm_y = palm_y

        # ----------------------------
        # VISUALS
        # ----------------------------

        cv2.circle(
            frame,
            (thumb_x, thumb_y),
            8,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            frame,
            (index_x, index_y),
            8,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            frame,
            (middle_x, middle_y),
            8,
            (255, 255, 0),
            -1
        )

        cv2.line(
            frame,
            (thumb_x, thumb_y),
            (index_x, index_y),
            (0, 255, 255),
            2
        )

        cv2.line(
            frame,
            (thumb_x, thumb_y),
            (middle_x, middle_y),
            (255, 0, 255),
            2
        )

        # ----------------------------
        # RIGHT CLICK
        # Thumb + Index + Middle together
        # ----------------------------

        right_click_gesture = (
            left_distance < RIGHT_CLICK_THRESHOLD
            and right_distance < RIGHT_CLICK_THRESHOLD
        )

        if (
            right_click_gesture
            and current_time - last_right_click > RIGHT_CLICK_DELAY
        ):

            pyautogui.rightClick()

            last_right_click = current_time

            pinch_start_time = None

            cv2.putText(
                frame,
                "RIGHT CLICK",
                (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                3
            )

        # ----------------------------
        # LEFT CLICK / DRAG
        # ----------------------------

        if (left_distance < CLICK_THRESHOLD) and (not right_click_gesture) and (not scroll_mode):

            if pinch_start_time is None:

                pinch_start_time = current_time

                drag_started = False

            pinch_duration = (
                current_time - pinch_start_time
            )

            if (
                pinch_duration > DRAG_HOLD_TIME
                and not dragging
            ):

                pyautogui.mouseDown()

                dragging = True
                drag_started = True

            if dragging:

                cv2.putText(
                    frame,
                    "DRAGGING",
                    (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

        else:

            if dragging:

                pyautogui.mouseUp()

                dragging = False

            elif (
                pinch_start_time is not None
                and not drag_started
            ):

                pinch_duration = (
                    current_time - pinch_start_time
                )

                if (
                    pinch_duration < DRAG_HOLD_TIME
                    and current_time - last_click > CLICK_DELAY
                ):

                    if current_time - last_release_click_time <= 0.4:

                        pyautogui.doubleClick()

                        cv2.putText(
                            frame,
                            "DOUBLE CLICK",
                            (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            3
                        )

                        last_release_click_time = 0

                    else:

                        pyautogui.click()

                        cv2.putText(
                            frame,
                            "CLICK",
                            (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            3
                        )

                        last_release_click_time = current_time

                    last_click = current_time

            pinch_start_time = None



        if current_time - last_screenshot_time < 1.5:

            cv2.putText(
                frame,
                "SCREENSHOT CAPTURED",
                (30, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                3
            )

        if scroll_mode:
            flash_state = not flash_state

            cv2.line(
                frame,
                (index_x, index_y),
                (middle_x, middle_y),
                (0, 255, 0),
                4
            )

            if flash_state:

                cv2.putText(
                    frame,
                    "SCROLL MODE",
                    (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

        cv2.putText(
            frame,
            f"L:{int(left_distance)}  R:{int(right_distance)}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    else:

        prev_palm_x = None
        prev_palm_y = None

        if dragging:

            pyautogui.mouseUp()

            dragging = False

        pinch_start_time = None

    cv2.putText(
        frame,
        f"FPS: {fps}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Q = Exit",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    if time.time() < screenshot_flash_until:
        frame[:] = 255

        cv2.putText(
            frame,
            "SCREENSHOT SAVED",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3
        )

    cv2.imshow(
        "Virtual Mouse",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()