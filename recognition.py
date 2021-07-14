import face_recognition
import cv2
import os

knownFile = face_recognition.load_image_file("tomar1.jpg")
knownImage = face_recognition.face_encodings(knownFile)[0]

directory = os.fsencode('images')
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    searchFile = face_recognition.load_image_file('images/' + filename)
    imageToSearch = face_recognition.face_encodings(searchFile)

    results = face_recognition.compare_faces(imageToSearch, knownImage)
    locations = face_recognition.face_locations(searchFile)
   
    cvImage = cv2.imread('images/' + filename)
    if not os.path.isdir('results'): os.mkdir('results')
    writeImage = False

    for i in range(len(results)):
        if results[i]:
            writeImage = True
            location = locations[i]
            cv2.rectangle(
                cvImage, 
                (location[3], location[0]), 
                (location[1], location[2]), 
                (255, 255, 0), 
                2
            )
    
    if writeImage: cv2.imwrite('results/' + 'result_' + filename, cvImage)