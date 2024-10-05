import numpy as np
import cv2

# Create a simple handwritten digit image
def create_handwritten_digit():
    # Create a black image
    image = np.zeros((100, 100), dtype=np.uint8)

    # Draw a digit (e.g., the digit '5') using cv2.putText
    cv2.putText(image, '5', (0, 105), cv2.FONT_HERSHEY_SIMPLEX, 5, (255), 10, cv2.LINE_AA)

    # Save the image
    cv2.imwrite('image.png', image)

create_handwritten_digit()
