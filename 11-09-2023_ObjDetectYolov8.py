import cv2
import argparse
import matplotlib.pyplot as plt
import supervision as sv
import numpy as np
import math

from ultralytics import YOLO

# Create a zone for detection for the whole window
ZONE_POLYGON = np.array([
    [0, 0],
    [1280, 0],
    [1280, 720],
    [0, 720]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Live")
    parser.add_argument("--webcam-resolution",
                        nargs=2,
                        type=int,
                        default=[1280, 720])
    args = parser.parse_args()
    return args


def calculate_distance(x1, y1, x2, y2):
    # Calculate the squared differences
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the squared distance
    squared_distance = dx**2 + dy**2

    # Calculate the distance by taking the square root
    distance = math.sqrt(squared_distance)

    print(distance)
    return distance


def line_and_angle(input, x1, y1, x2, y2, distance):
    # Draw line between point
    cv2.line(input, (x1, y1), (x2, y2), (0, 0, 255), 2)

    text_x = (x1 + x2) // 2
    text_y = (y1 + y2) // 2 - 10

    # Calculate degrees
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)

    cv2.putText(input, f"{distance},  {angle_deg:.2f}", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)


def click_event(event, x, y, flags, params):
    global clicked_points
    global exception_poly
    global clicked_axis
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # appending the clicked point to the list
        clicked_points.append((x, y))
        # print(f"{x}, {y}")

        if clicked_axis == "":
            clicked_axis = f"[{x}, {y}]"

        else:
            clicked_axis += f", [{x}, {y}]"
    elif event == cv2.EVENT_RBUTTONDOWN:
        exception_poly.append((x, y))



def create_shape_and_cut():
    global exception_poly
    global frame

    # Create a mask for the shape
    mask = np.zeros_like(frame, dtype=np.uint8)
    exception_polys = np.array(exception_poly)

    # Draw the shape (assuming it's a rectangle)
    cv2.fillPoly(mask, [exception_polys], (255, 255, 255))

    # Apply the mask to the frame to "cut" the shape
    result = cv2.bitwise_and(frame, mask)

    # Invert the mask to keep the area outside the shape
    inverted_mask = cv2.bitwise_not(mask)
    outside_roi = cv2.bitwise_and(frame, inverted_mask)
    return outside_roi


args = parse_arguments()
frame_width, frame_height = args.webcam_resolution

clicked_points = []
exception_poly = []
clicked_axis = ""
custom_poly = True

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

model = YOLO("yolov8n.pt")

box_annotator = sv.BoxAnnotator(thickness=2,
                                text_thickness=2,
                                text_scale=1)

zone = sv.PolygonZone(polygon=ZONE_POLYGON,
                      frame_resolution_wh=tuple(args.webcam_resolution))
zone_annotator = sv.PolygonZoneAnnotator(zone=zone,
                                         color=sv.Color.red(),
                                         text_padding=0)

while True:
    ret, frame = cap.read()
    scene = frame

    if len(clicked_points) >= 3 or len(exception_poly) >= 3:
        if custom_poly is False:
            # ZONE_POLYGON_2 = np.array(eval(clicked_axis))
            # zone2 = sv.PolygonZone(polygon=ZONE_POLYGON_2,
            #                        frame_resolution_wh=tuple(
            #                            args.webcam_resolution))
            #
            # zone2_annotator = sv.PolygonZoneAnnotator(zone=zone2,
            #                                           color=sv.Color.blue(),
            #                                           text_padding=0)
            # zone_trigger = zone2.trigger(detections=detections)
            #
            # frame = zone2_annotator.annotate(scene=frame)

            frame2 = create_shape_and_cut()
            exception_polys = np.array(exception_poly)
            exept = np.zeros(frame, dtype=np.uint8)
            color = (0, 255, 0, 128)
            cv2.fillPoly(frame, [exception_polys], color)
            frame = cv2.addWeighted(exept, 0.5, frame, 0.5, 0)
            scene = frame2

        elif custom_poly is True:
            if (cv2.waitKey(30) == 32):
                custom_poly = False
    M = 0
    centrolist = []

    # Custom polygon based on axis points created by clicking the window
    result = model.track(scene, persist=True, agnostic_nms=True)[0]
    # result = model.predict(frame,agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(result)
    # detections = detections[detections.class_id == 0]
    labels = [
        f"#{track_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, track_id
        in detections
    ]
    cam = box_annotator.annotate(scene=frame,
                                 detections=detections,
                                 labels=labels)

    zone_trigger = zone.trigger(detections=detections)
    frame = zone_annotator.annotate(scene=frame)



    # Generate a circle at the exact location where a click occurs.
    for point in clicked_points:
        cv2.circle(frame, point, 5, (255, 0, 0),     -1)
    for point in exception_poly:
        cv2.circle(frame, point, 5, (0, 255, 0),     -1)

    if len(zone_trigger) != 0:
        dxy = detections.xyxy

        print(f"Detected Object = {len(zone_trigger)}")

        # if len(zone_trigger) > 2:
        #     xcentro = (dxy[2, 2] - dxy[2, 0]) / 2 + dxy[2, 0]
        #     print(xcentro)

        for x in range(0, len(zone_trigger)):
            print(f"Current Loop = {M}")
            xcentro = (dxy[M, 2] - dxy[M, 0]) / 2 + dxy[M, 0]
            ycentro = (dxy[M, 3] - dxy[M, 1]) / 2 + dxy[M, 1]

            centrolist.append(xcentro)
            centrolist.append(ycentro)

            print(f"X centroids = {xcentro}, Y centroids = {ycentro}")

            M = M + 1
        while len(centrolist) != 2:
            m, n, x, y = 0, 1, 2, 3
            while len(centrolist) >= y:
                centro_distance = calculate_distance(centrolist[m], centrolist[n],
                                                     centrolist[x], centrolist[y])

                line_and_angle(frame,
                               int(centrolist[m]), int(centrolist[n]),
                               int(centrolist[x]), int(centrolist[y]),
                               int(centro_distance))

                x = x + 2
                y = y + 2
            centrolist = centrolist[2:]

    print(f"Total Clicked Point = {len(clicked_points)}")
    print((f"Exection Poly = {len(exception_poly)}"))

    cv2.imshow("yolov8", cam)
    cv2.setMouseCallback('yolov8', click_event)

    if (cv2.waitKey(30) == 27):
        break