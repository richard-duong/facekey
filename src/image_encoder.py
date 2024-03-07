import os
import numpy as np
import face_recognition
import pickle

RESOURCE_DIR = "resources"


def create_face_encodings_from_images(root_directory):
    person_encodings = {}

    for person_name in os.listdir(root_directory):
        person_path = os.path.join(root_directory, person_name)
        print("person_path: ", person_path)

        if os.path.isdir(person_path):
            encodings = []

            for image_file in os.listdir(person_path):
                if image_file.endswith(".jpg"):
                    image_path = os.path.join(person_path, image_file)
                    # Load the image
                    image = face_recognition.load_image_file(image_path)
                    # Detect faces in the image
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        # Assuming each image contains only one face
                        encoding = face_encodings[0]
                        encodings.append(encoding)

            # Store the encodings for this person
            if encodings:
                person_encodings[person_name] = np.array(encodings)

    return person_encodings


def main():
    encodings = create_face_encodings_from_images(RESOURCE_DIR)
    # Saving the encodings
    with open("face_encodings.pkl", "wb") as f:
        pickle.dump(encodings, f)


if __name__ == "__main__":
    main()
