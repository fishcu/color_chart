import cv2

cv2.namedWindow('Press any key')

# Wait for a key press and store the result in a variable
key_pressed = cv2.waitKeyEx(0)

# Mask out anything but the last 8 bits to get the ASCII value
key_pressed_ascii = key_pressed & 0xFF

# Convert the ASCII value to a character
key_char = chr(key_pressed_ascii)

print(f"The key pressed is: {key_char} (ASCII: {key_pressed_ascii})")
