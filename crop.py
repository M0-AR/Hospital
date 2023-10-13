import cv2
def crop_image(image_path, annotation_path):
    with open(annotation_path, 'r') as file:
        line = file.readline().strip()
        _, x1, y1, x2, y2 = line.split()  

    x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image at path: {image_path}")
        return

    # Denormalize if needed, and check for validity
    if all(0 <= coord <= 1 for coord in [x1, y1, x2, y2]):
        h, w, _ = img.shape
        x1, x2 = [int(coord * w) for coord in [x1, x2]]
        y1, y2 = [int(coord * h) for coord in [y1, y2]]
        x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    # Check validity of coordinates
    if x2 <= x1 or y2 <= y1:
        print(f"Invalid cropping coordinates: {x1, y1, x2, y2}")
        return

    cropped_img = img[y1:y2, x1:x2]

    if cropped_img.size == 0:
        print(f"Failed to crop image with coordinates: {x1, y1, x2, y2}")
        return

    return cropped_img



# Read the image and annotation back and crop the image
img_path = './frames/272.jpg' 
annot_path = './frames/272.txt'
cropped_img = crop_image(img_path, annot_path)

# Display the cropped image
cv2.imshow('Cropped Image', cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()