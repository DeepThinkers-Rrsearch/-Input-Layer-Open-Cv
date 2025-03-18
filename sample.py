# import cv2
# import numpy as np

# # Load FSM image
# image = cv2.imread("1.png")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply Gaussian Blur to reduce noise
# blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# # Detect circles (states) using HoughCircles with adjusted minRadius
# circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 30,
#                            param1=100, param2=30, minRadius=20, maxRadius=60)  # Adjust minRadius

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
        
#         # Check for double circles (final states)
#         roi = gray[y - r - 5:y + r + 5, x - r - 5:x + r + 5]
#         roi_blurred = cv2.GaussianBlur(roi, (5, 5), 2)
#         inner_circles = cv2.HoughCircles(roi_blurred, cv2.HOUGH_GRADIENT, 1, 10,
#                                          param1=100, param2=20, minRadius=r - 5, maxRadius=r + 5)

#         is_final = inner_circles is not None and len(inner_circles[0]) > 1

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

import cv2
import numpy as np

# Load FSM image
image = cv2.imread("1.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles (states) using HoughCircles with adjusted minRadius
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 30,
                           param1=100, param2=30, minRadius=20, maxRadius=60)  # Adjusted minRadius

# Copy the image for drawing
output = image.copy()
filtered_circles = []
final_states = []

if circles is not None:
    circles = np.uint16(np.around(circles))

    for i, circle in enumerate(circles[0, :]):
        x, y, r = circle

        # Ignore very small circles (likely self-loops)
        if r < 25:  # Self-loops are typically smaller, adjust as needed
            continue
        
        # Expand ROI for better detection
        roi_padding = int(r * 1.2)  # Expanding the region
        roi = gray[max(y - roi_padding, 0):min(y + roi_padding, gray.shape[0]),
                   max(x - roi_padding, 0):min(x + roi_padding, gray.shape[1])]
        
        # Apply minimal blur to retain details
        roi_blurred = cv2.GaussianBlur(roi, (3, 3), 1)

        # Debug: Show extracted ROI
        # cv2.imshow(f"ROI {i}", roi)
        # cv2.waitKey(0)

        # Detect inner circles (final states)
        inner_circles = cv2.HoughCircles(roi_blurred, cv2.HOUGH_GRADIENT, 1, 10,
                                         param1=100, param2=15, minRadius=r - 10, maxRadius=r + 15)

        is_final = inner_circles is not None and len(inner_circles[0]) > 1 if inner_circles is not None else False

        # Draw the state outline
        color = (255, 0, 0) if is_final else (0, 255, 0)  # Blue for final state, Green otherwise
        cv2.circle(output, (x, y), r, color, 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Mark center

        # Store detected states
        filtered_circles.append((x, y, r))
        if is_final:
            final_states.append((x, y, r))

    print(f"Detected {len(filtered_circles)} states.")
    print(f"Detected {len(final_states)} final states.")
else:
    print("No states detected. Try adjusting minRadius/maxRadius.")

# Show and save the result
cv2.imshow("FSM States", output)
cv2.imwrite("fsm_states_filtered.png", output)
cv2.waitKey(0)
cv2.destroyAllWindows()




# from google.cloud import vision

# client = vision.ImageAnnotatorClient()

# with open("1.png", "rb") as image_file:
#     content = image_file.read()

# image = vision.Image(content=content)
# response = client.text_detection(image=image)
# texts = response.text_annotations

# for text in texts:
#     print(f"Detected text: {text.description}")

