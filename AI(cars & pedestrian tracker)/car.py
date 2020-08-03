import cv2

# Our Image
img_file = 'image.png'
#video = cv2.VideoCapture('tesla.mp4')
video = cv2.VideoCapture('both.mp4')

# Our pre-trained car & pedestrain classifier
car_tracker_file = 'car_detection.xml'
pedestrian_tracker_file = 'persons_detection.xml'
# create car clasifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# Run forever until car stops or something happens
while True:
    # Read the current frame
    (read_successful,frame) = video.read()
    # safe coding
    if read_successful:
        #Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # Draw  rectangles around the cars

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+2, y+3), (x + w, y + h), (255, 0, ), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Draw  rectangles around the pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Display the image with faces spotted

    cv2.imshow('charan AI', frame)

    # Don't autoclose(Wait here and listen for a key press)

    key = cv2.waitKey(3)

    # stop if Q or q is pressed
    if key==81 or key==113:
        break


"""
# create opencv image
img = cv2.imread(img_file)

# convert to grayscale (needed for haar cascade)
black_n_white = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# create car clasifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Detect cars
cars = car_tracker.detectMultiScale(black_n_white)

# Draw  rectangles around the cars

for (x,y,w,h) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

# Display the image with faces spotted

cv2.imshow('charan AI',img)

# Don't autoclose(Wait here and listen for a key press)

cv2.waitKey()

print("hello cherry")
"""