# # import cv2
# # import numpy as np

# # # Load FSM image
# # image = cv2.imread("1.png")
# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # # Apply Gaussian Blur to reduce noise
# # blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# # # Detect circles (states) using HoughCircles with adjusted minRadius
# # circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 30,
# #                            param1=100, param2=30, minRadius=20, maxRadius=60)  # Adjust minRadius

# # # Copy the image for drawing
# # output = image.copy()
# # filtered_circles = []
# # final_states = []

# # if circles is not None:
# #     circles = np.uint16(np.around(circles))
    
# #     for i, circle in enumerate(circles[0, :]):
# #         x, y, r = circle

# #         # Ignore very small circles (likely self-loops)
# #         if r < 25:  # Self-loops are typically smaller, adjust as needed
# #             continue
        
# #         # Check for double circles (final states)
# #         roi = gray[y - r - 5:y + r + 5, x - r - 5:x + r + 5]
# #         roi_blurred = cv2.GaussianBlur(roi, (5, 5), 2)
# #         inner_circles = cv2.HoughCircles(roi_blurred, cv2.HOUGH_GRADIENT, 1, 10,
# #                                          param1=100, param2=20, minRadius=r - 5, maxRadius=r + 5)

# #         is_final = inner_circles is not None and len(inner_circles[0]) > 1

# #         # Draw the state outline
# #         color = (255, 0, 0) if is_final else (0, 255, 0)  # Blue for final state, Green otherwise
# #         cv2.circle(output, (x, y), r, color, 2)
# #         cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Mark center

# #         # Store detected states
# #         filtered_circles.append((x, y, r))
# #         if is_final:
# #             final_states.append((x, y, r))

# #     print(f"Detected {len(filtered_circles)} states.")
# #     print(f"Detected {len(final_states)} final states.")
# # else:
# #     print("No states detected. Try adjusting minRadius/maxRadius.")

# # # Show and save the result
# # cv2.imshow("FSM States", output)
# # cv2.imwrite("fsm_states_filtered.png", output)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Load FSM image
# image = cv2.imread(r"C:\Users\Shyamjith Jayasinghe\Documents\GitHub\-Input-Layer-Open-Cv\1.png")

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply Gaussian Blur to reduce noise
# blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# # Detect circles (states) using HoughCircles with adjusted minRadius
# circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 30,
#                            param1=100, param2=30, minRadius=20, maxRadius=60)  # Adjusted minRadius

# # Copy the image for drawing
# output = image.copy()
# filtered_circles = []
# final_states = []

# if circles is not None:
#     circles = np.uint16(np.around(circles))

#     for i, circle in enumerate(circles[0, :]):
#         x, y, r = circle

#         # Ignore very small circles (likely self-loops)
#         if r < 25:  # Self-loops are typically smaller, adjust as needed
#             continue
        
#         # Expand ROI for better detection
#         roi_padding = int(r * 1.2)  # Expanding the region
#         roi = gray[max(y - roi_padding, 0):min(y + roi_padding, gray.shape[0]),
#                    max(x - roi_padding, 0):min(x + roi_padding, gray.shape[1])]
        
#         # Apply minimal blur to retain details
#         roi_blurred = cv2.GaussianBlur(roi, (3, 3), 1)

#         # Debug: Show extracted ROI
#         # cv2.imshow(f"ROI {i}", roi)
#         # cv2.waitKey(0)

#         # Detect inner circles (final states)
#         inner_circles = cv2.HoughCircles(roi_blurred, cv2.HOUGH_GRADIENT, 1, 10,
#                                          param1=100, param2=15, minRadius=r - 10, maxRadius=r + 15)

#         is_final = inner_circles is not None and len(inner_circles[0]) > 1 if inner_circles is not None else False

#         # Draw the state outline
#         color = (255, 0, 0) if is_final else (0, 255, 0)  # Blue for final state, Green otherwise
#         cv2.circle(output, (x, y), r, color, 2)
#         cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Mark center

#         # Store detected states
#         filtered_circles.append((x, y, r))
#         if is_final:
#             final_states.append((x, y, r))

#     print(f"Detected {len(filtered_circles)} states.")
#     print(f"Detected {len(final_states)} final states.")
# else:
#     print("No states detected. Try adjusting minRadius/maxRadius.")

# # Show and save the result
# cv2.imshow("FSM States", output)
# cv2.imwrite("fsm_states_filtered.png", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# # from google.cloud import vision

# # client = vision.ImageAnnotatorClient()

# # with open("1.png", "rb") as image_file:
# #     content = image_file.read()

# # image = vision.Image(content=content)
# # response = client.text_detection(image=image)
# # texts = response.text_annotations

# # for text in texts:
# #     print(f"Detected text: {text.description}")

#---------------------------------------------------------------------------------------------
# import cv2
# import numpy as np

# # Load the FSM image
# image = cv2.imread(r'C:\Users\Shyamjith Jayasinghe\Documents\GitHub\-Input-Layer-Open-Cv\1.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# output = image.copy()

# # Preprocessing
# blurred = cv2.GaussianBlur(gray, (9, 9), 2)
# edges = cv2.Canny(blurred, 50, 150)

# # Detect states (circles)
# circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 30,
#                            param1=100, param2=30, minRadius=20, maxRadius=60)

# states = []
# final_states = []

# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i, circle in enumerate(circles[0, :]):
#         x, y, r = circle
#         if r < 25:
#             continue
#         roi_padding = int(r * 1.2)
#         roi = gray[max(y - roi_padding, 0):min(y + roi_padding, gray.shape[0]),
#                    max(x - roi_padding, 0):min(x + roi_padding, gray.shape[1])]
#         roi_blurred = cv2.GaussianBlur(roi, (3, 3), 1)

#         # Detect possible double circle
#         inner_circles = cv2.HoughCircles(roi_blurred, cv2.HOUGH_GRADIENT, 1, 10,
#                                          param1=100, param2=15, minRadius=r - 10, maxRadius=r + 15)
#         is_final = inner_circles is not None and len(inner_circles[0]) > 1 if inner_circles is not None else False

#         color = (255, 0, 0) if is_final else (0, 255, 0)  # Blue - final, Green - normal
#         cv2.circle(output, (x, y), r, color, 2)
#         cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

#         states.append((x, y, r))
#         if is_final:
#             final_states.append((x, y, r))

# print(f"Detected {len(states)} states.")
# print(f"Detected {len(final_states)} final states.")

# # ---- Arrow Detection ----
# arrows = []
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if area < 300:  # filter out small noise
#         continue

#     approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
#     x, y, w, h = cv2.boundingRect(cnt)

#     aspect_ratio = w / float(h)

#     if 0.5 < aspect_ratio < 2.0:
#         shape = "Straight Arrow"
#     else:
#         shape = "Curved Arrow"

#     arrows.append(((x, y, w, h), shape))
#     cv2.drawContours(output, [cnt], -1, (0, 255, 255), 2)
#     cv2.putText(output, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 255), 1)

# print(f"Detected {len(arrows)} arrows (with straight/curved classification).")

# # ---- Initial State Detection ----
# incoming = {i: 0 for i in range(len(states))}
# outgoing = {i: 0 for i in range(len(states))}

# for idx, (x, y, w, h) in enumerate([arrow[0] for arrow in arrows]):
#     arrow_center = (x + w // 2, y + h // 2)
#     for i, (sx, sy, r) in enumerate(states):
#         distance = np.hypot(arrow_center[0] - sx, arrow_center[1] - sy)
#         if distance < r + 30:
#             incoming[i] += 1

# for i in range(len(states)):
#     if incoming[i] == 0:
#         print(f"State q{i} might be the Initial State (No incoming arrows).")
#         cv2.putText(output, "Initial", (states[i][0] - 30, states[i][1] - 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# # ---- Final Visualization ----
# cv2.imshow("FSM Detection", output)
# cv2.imwrite("fsm_detection_result.png", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#---------------------------------------------------------------------------------------------

import cv2
import numpy as np

# Structure to store state information
class State:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.is_final = False
        self.is_initial = False

# Load the image
#image_path = "1.png"  # Change this to your image path
image = cv2.imread(r'C:\Users\Shyamjith Jayasinghe\Documents\GitHub\-Input-Layer-Open-Cv\5.png')


if image is None:
    print("Could not open or find the image.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold to get a binary image
#_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)


# Detect circles (FSM states)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                           param1=100, param2=30,
                           minRadius=20, maxRadius=40)

states = []
if circles is not None:
    circles = np.uint16(np.around(circles))
    for c in circles[0, :]:
        states.append(State((c[0], c[1]), c[2]))

# # Detect inner circles (potential final states)
# inner_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 5,
#                                  param1=100, param2=30,
#                                  minRadius=10, maxRadius=20)

# if inner_circles is not None:
#     inner_circles = np.uint16(np.around(inner_circles))
#     for state in states:
#         for inner in inner_circles[0, :]:
#             inner_center = (inner[0], inner[1])
#             inner_radius = inner[2]
#             if np.linalg.norm(np.array(state.center) - np.array(inner_center)) < 5 and inner_radius < state.radius:
#                 state.is_final = True
#                 break
# Detect potential inner circles (used for detecting final states)
# # Detect potential inner circles (used for detecting final states)
# inner_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10,
#                                  param1=100, param2=30,
#                                  minRadius=10, maxRadius=25)  # Adjust maxRadius if needed
#   # Adjusted maxRadius

# # Map detected inner circles to their respective states
# inner_circle_count = {i: 0 for i in range(len(states))}

# if inner_circles is not None:
#     inner_circles = np.uint16(np.around(inner_circles))
#     for inner in inner_circles[0, :]:
#         inner_center = (inner[0], inner[1])
#         inner_radius = inner[2]
        
#         # Find the closest state (outer circle)
#         nearest_state_idx = -1
#         min_dist = float('inf')

#         for i, state in enumerate(states):
#             dist = np.linalg.norm(np.array(state.center) - np.array(inner_center))
#             if dist < 5 and inner_radius < state.radius:
#                 nearest_state_idx = i
#                 min_dist = dist
        
#         if nearest_state_idx != -1:
#             inner_circle_count[nearest_state_idx] += 1  # Count inner circles for each state

# # Mark states as final if they contain 2 inner circles
# for i, state in enumerate(states):
#     if inner_circle_count[i] >= 2:  # Ensure at least two nested circles
#         state.is_final = True
#         cv2.circle(image, state.center, state.radius - 5, (0, 255, 0), 2)  # Draw extra final state indicator


# Detect potential outer circles (used for detecting final states)
# outer_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10,
#                                  param1=100, param2=30,
#                                  minRadius=40, maxRadius=70)  # Adjust min/max radius based on outer ring size
# Apply Gaussian blur to smooth the image
blurred = cv2.GaussianBlur(gray, (5, 5), 1)

# Detect potential outer circles (used for detecting final states)
outer_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 10,
                                 param1=100, param2=20,  # Lower param2 for more detections
                                 minRadius=42, maxRadius=61)  # Adjust min/max radius based on outer ring size


# Map detected outer circles to their respective states
outer_circle_count = {i: 0 for i in range(len(states))}

if outer_circles is not None:
    outer_circles = np.uint16(np.around(outer_circles))
    for outer in outer_circles[0, :]:
        outer_center = (outer[0], outer[1])
        outer_radius = outer[2]

        # Find the closest state (inner circle) to this outer circle
        nearest_state_idx = -1
        min_dist = float('inf')

        for i, state in enumerate(states):
            dist = np.linalg.norm(np.array(state.center) - np.array(outer_center))
            # Check if the outer circle is around the state
            if dist < 5 and outer_radius > state.radius:
                nearest_state_idx = i
                min_dist = dist

        if nearest_state_idx != -1:
            outer_circle_count[nearest_state_idx] += 1  # Count outer circles for each state

# Mark states as final if they have a corresponding outer circle
for i, state in enumerate(states):
    if outer_circle_count[i] >= 1:  # One outer circle is enough
        state.is_final = True
        cv2.circle(image, state.center, state.radius + 5, (0, 255, 0), 2)  # Optional: Highlight the final state

# Detect transitions (lines)
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

transitions = []
outgoing = [0] * len(states)
incoming = [0] * len(states)

# Function to find the nearest state to a point
def find_nearest_state(point):
    min_dist = float('inf')
    nearest_idx = -1
    for i, state in enumerate(states):
        dist = np.linalg.norm(np.array(state.center) - np.array(point))
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i
    return nearest_idx

if lines is not None:
    for l in lines:
        x1, y1, x2, y2 = l[0]
        start = (x1, y1)
        end = (x2, y2)

        from_state = find_nearest_state(start)
        to_state = find_nearest_state(end)

        if from_state != -1 and to_state != -1:
            transitions.append((from_state, to_state))
            outgoing[from_state] += 1
            incoming[to_state] += 1
            cv2.line(image, start, end, (0, 0, 255), 2)

# Detect initial states (states with no incoming edges but with outgoing edges)
for i, state in enumerate(states):
    if incoming[i] == 0 and outgoing[i] > 0:
        state.is_initial = True
        cv2.circle(image, state.center, state.radius + 5, (255, 255, 0), 2)

# Detect self-loops
binary_loops = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
contours, _ = cv2.findContours(binary_loops, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

self_loop_count = 0
for contour in contours:
    if len(contour) > 5:
        ellipse = cv2.fitEllipse(contour)
        ellipse_center = (int(ellipse[0][0]), int(ellipse[0][1]))

        nearest_state_idx = find_nearest_state(ellipse_center)
        if nearest_state_idx != -1:
            distance = np.linalg.norm(np.array(states[nearest_state_idx].center) - np.array(ellipse_center))
            if states[nearest_state_idx].radius * 0.5 < distance < states[nearest_state_idx].radius * 1.5:
                cv2.ellipse(image, ellipse, (255, 0, 255), 2)
                self_loop_count += 1

# Draw detected states
for state in states:
    cv2.circle(image, state.center, state.radius, (0, 255, 0), 2)
    if state.is_final:
        cv2.circle(image, state.center, state.radius - 5, (0, 255, 0), 2)

# Print detected components
# print(f"Detected States: {len(states)}")
# print("Final States:", [(s.center[0], s.center[1]) for s in states if s.is_final])
# print("Initial States:", [(s.center[0], s.center[1]) for s in states if s.is_initial])
# print(f"Detected Transitions: {len(transitions)}")
# print(f"Total self-loops detected: {self_loop_count}")


# Print detected components
print(f"Detected States: {len(states)}")
print(f"Number of Final States: {sum(1 for s in states if s.is_final)}")
print(f"Number of Initial States: {sum(1 for s in states if s.is_initial)}")
print("Final States Coordinates:", [(s.center[0], s.center[1]) for s in states if s.is_final])
print("Initial States Coordinates:", [(s.center[0], s.center[1]) for s in states if s.is_initial])
print(f"Detected Transitions: {len(transitions)}")
print(f"Total self-loops detected: {self_loop_count}")


# Show the result
cv2.imshow("FSM Detection", image)
cv2.imwrite("fsm_detection_python.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
