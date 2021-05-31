import face_alignment
import numpy as np
from skimage import io
import os
from PIL import Image, ImageDraw
from scipy.interpolate import interp1d

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cpu')


RAW_IMAGES_DIR = '../stylegan_data/face_dataset_cropped_mark3/'
SAVE_IMAGES_DIR = '../pix2pix_data/A/'
SAVE_IMAGES_DIR2 = '../pix2pix_data/B/'
PHONEME_IMAGES_DIR = '../pix2pix_data/phoneme_images/'

# EYES = 36 - 48
# MOUTH = 48 - 68

def create_face_map(filename, counter):
    # preds = fa.get_landmarks_from_directory(RAW_IMAGES_DIR)

    input = io.imread(RAW_IMAGES_DIR+filename)
    preds = fa.get_landmarks(input)
    image = Image.open(RAW_IMAGES_DIR+filename)

    mask = Image.new("RGBA", (512, 512), (255, 255, 255))

    draw = ImageDraw.Draw(mask)
    for pnts in preds[0]:
        draw.point((pnts[0], pnts[1]), fill="green")
    # mask.show()

    image = image.resize((512, 512))
    mask.save(SAVE_IMAGES_DIR + counter + '.png')
    image.save(SAVE_IMAGES_DIR2 + counter + '.png')
    print()


def store_mouth_vectors():

    for filename in [f for f in os.listdir(PHONEME_IMAGES_DIR) if f[0] not in '._']:

        input = io.imread(PHONEME_IMAGES_DIR+filename)
        preds = fa.get_landmarks(input)
        mouth_vector = preds[0]
        newfilename = filename.split('_')[-1].split('.')[0]
        np.savetxt('../phoneme_csv/{}.csv'.format(newfilename), mouth_vector, delimiter=',')



def create_dataset():
    counter = 0
    for filename in [f for f in os.listdir(RAW_IMAGES_DIR) if f[0] not in '._']:

        if counter > 1920:
            create_face_map(filename, str(counter))
            print(counter)
        counter += 1

if __name__ == "__main__":
    # create_dataset()
    store_mouth_vectors()