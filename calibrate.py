"""
One-time calibration for speed estimation.

Click 4 points on a frame that form a rectangle ON THE ROAD SURFACE, in order:
  top-left, top-right, bottom-right, bottom-left.
Good choices: lane edges + two dashed-line marks a known distance apart.

You must know the REAL dimensions of that rectangle in meters:
  - lane width (US/EU highway lane ~= 3.5 m)
  - length along the road between your top and bottom edges (measure via
    dashed lane markings: in the US a dash+gap cycle is ~12 m; EU ~= 15 m)

Outputs the SRC_POINTS and REAL_W / REAL_H to paste into fingerprint_video_speed.py.

Run: python calibrate.py input.mp4
"""
import sys
import cv2

pts = []


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append((x, y))
        print(f"point {len(pts)}: ({x}, {y})")


def main(source):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("cannot read frame")

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    print("Click 4 road points: TL, TR, BR, BL. Press any key when done.")

    while True:
        disp = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(disp, p, 5, (0, 0, 255), -1)
            cv2.putText(disp, str(i + 1), (p[0] + 6, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(pts) == 4:
            cv2.polylines(disp, [__import__("numpy").array(pts)], True,
                          (0, 255, 0), 2)
        cv2.imshow("calibrate", disp)
        if cv2.waitKey(20) != -1 and len(pts) == 4:
            break

    cv2.destroyAllWindows()
    print("\nPaste into fingerprint_video_speed.py:")
    print(f"SRC_POINTS = {pts}")
    print("REAL_W = 3.5     # <-- set to the real width  (meters) of your rectangle")
    print("REAL_H = 24.0    # <-- set to the real length (meters) of your rectangle")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "input.mp4")
