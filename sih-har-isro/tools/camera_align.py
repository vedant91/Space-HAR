"""
Camera alignment preview — nothing to do with the HAR pipeline itself.
Opens the given camera and shows the raw feed so you can position a phone
(or the real rack) in frame before starting the real, headless pipeline run.

Usage:
    python tools/camera_align.py            # camera 0
    python tools/camera_align.py --camera 1

Press Q or Esc to close.
"""
import argparse
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}. Try --camera 1.")
        return

    print("Position your phone/rack in frame. Press Q or Esc to close.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w // 2, h // 2), (0, 255, 0),
                        markerType=cv2.MARKER_CROSS, markerSize=30, thickness=1)
        cv2.putText(frame, "Align phone here - Q to close", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Camera alignment", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
