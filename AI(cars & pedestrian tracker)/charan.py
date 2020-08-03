import cv2

# Our Image
img_file = 'image.png'

# Our pre-trained car classifier
classifier_file = 'car_detection.xml'


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