import os
# import face_recognition
from PIL import Image
# cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade

RAW_IMAGES_DIR = os.path.expanduser('/home/willreynolds/Documents/GitHub/alignment_tools/face_dataset_phonemes_cropped_mark3/')
SAVE_IMAGES_DIR = os.path.expanduser("../face_dataset_phonemes_cropped_mark3/")


# def align():
#
#     for file in [f for f in os.listdir(RAW_IMAGES_DIR) if f[0] not in '._']:
#
#         image_path = RAW_IMAGES_DIR+file
#         image = Image.open(image_path)
#         image_req = face_recognition.load_image_file(image_path)
#         face_locations = face_recognition.face_locations(image_req)[0]
#         top = face_locations[0]
#         right = face_locations[1]
#         bottom = face_locations[2]
#         left = face_locations[3]
#         image = image.crop((left, top, right, bottom))
#         # image = image.crop(face_locations)
#         image = image.resize((512, 512))
#         image.save(SAVE_IMAGES_DIR+file)
#         print()


def to_png():

    counter = 0
    for file in [f for f in os.listdir(RAW_IMAGES_DIR) if f[0] not in '._']:
        if file[-4:] == '.png':
            image_path = RAW_IMAGES_DIR+file
            image = Image.open(image_path)

            bg = Image.new("RGBA", (512, 512), (255, 255, 255))
            bg.paste(image, (0, 0), image)
            new_image_path = RAW_IMAGES_DIR + file[:-4] + '.jpg'
            bg = bg.convert('RGB')
            bg.save(new_image_path)

            if file[:-4] == '.jpg':
                os.remove(image_path)

            counter += 1
            if counter % 50 == 0:
                print('Converted {}'.format(counter))

to_png()
