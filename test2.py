import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quiting..")
        break

cap.release()
cv2.destroyAllWindows()






# image = cv2.imread("img.png")

# pt1 = (120, 20)
# pt2 = (15, 10)
# color = (200, 0, 0)
# thickness = 4

# cv2.rectangle(image, pt1, pt2, color, thickness)

# cv2.imshow("Line", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cropped = image[100:200, 50:150]

# gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

# cv2.imshow("Cropped", gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# if image is None:
#     print("Error")
# else:
#     resized = cv2.resize(image, (300, 300)) # (width, height)
#     cv2.imshow("r", resized)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



# if image is not None:
#     h, w, c = image.shape
#     print(f"Height: {h}\nWidth: {w}\nColor: {c}")
# else:
#     print("Error")


# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Grey Scale", gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

