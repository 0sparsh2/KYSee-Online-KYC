import cv2

# Initialize Webcam
cap = cv2.VideoCapture(0)
import pkg_resources

#Load Haarcascade Frontal Face Classifier
xmlfile = pkg_resources.resource_filename(
    'cv2', 'data/haarcascade_frontalface_default.xml')
face_classifier = cv2.CascadeClassifier(xmlfile)

#face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Function returns cropped face
def face_extractor(photo):
    gray_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_photo)
    
    if faces is ():
        return None
    
    else:
        # Crop all faces found
        for (x,y,w,h) in faces:
            cropped_face = photo[y:y+h, x:x+w]
        
        return cropped_face


count = 0

# Collect 100 samples of your face from webcam input
while True:
    status,photo = cap.read()
    
    if face_extractor(photo) is not None:
        count += 1
        face = cv2.resize(face_extractor(photo), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name (Here, I am training for 2 members)
        file_name_path = '/Users/Anil/python codes/coduh2/myself/me/' + str(count) + '.jpg'
        #file_name_path = '/home/chiraggl/tlfr/faces/test/chirag/face' + str(count) + '.jpg'
        
        #file_name_path = '/home/chiraggl/tlfr/faces/train/ashwani/face' + str(count) + '.jpg'
        #file_name_path = '/home/chiraggl/tlfr/faces/test/ashwani/face' + str(count) + '.jpg'
        
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        pass

    if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
    #if cv2.waitKey(1) == 13 or count == 70: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()