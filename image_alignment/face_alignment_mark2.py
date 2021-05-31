import face_alignment
import numpy as np
from skimage import io
import os
from PIL import Image, ImageDraw

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cpu')


RAW_IMAGES_DIR = '../stylegan_data/face_dataset_mark2/'
SAVE_IMAGES_DIR = '../stylegan_data/face_dataset_cropped_mark4/'

def align_faces_top_botton():
    # preds = fa.get_landmarks_from_directory(RAW_IMAGES_DIR)
    for filename in [f for f in os.listdir(RAW_IMAGES_DIR) if f[0] not in '._']:

        input = io.imread(RAW_IMAGES_DIR+filename)
        preds = fa.get_landmarks(input)
        image = Image.open(RAW_IMAGES_DIR+filename)

        y = preds[0][:, 1]
        x = preds[0][:, 0]
        width, height = image.size

        top = height - preds[0][list(y).index(max(y))][1]
        bottom = height - preds[0][list(y).index(min(y))][1]
        right = preds[0][list(x).index(max(x))][0]
        left = preds[0][list(x).index(min(x))][0]

        diff = ((right - left) - (bottom - top)) / 2
        top = top - diff
        bottom = bottom + diff

        image = image.crop((left, top, right, bottom))
        image = image.resize((512, 512))
        image.save(SAVE_IMAGES_DIR + filename.split('_')[-1])
        print()


def align_faces_average_center():
    # preds = fa.get_landmarks_from_directory(RAW_IMAGES_DIR)
    for filename in [f for f in os.listdir(RAW_IMAGES_DIR) if f[0] not in '._']:
        input = io.imread(RAW_IMAGES_DIR + filename)
        preds = fa.get_landmarks(input)
        image = Image.open(RAW_IMAGES_DIR + filename)

        # draw = ImageDraw.Draw(image)
        # for pnts in preds[0]:
        #     draw.point((pnts[0], pnts[1]), fill="green")
        # image.save('out.png')
        box_radius = 300

        y = preds[0][:, 1]
        x = preds[0][:, 0]
        width, height = image.size
        avg_y = np.mean(y)
        avg_x = np.mean(x)

        top = (height - avg_y) - (box_radius) - (box_radius * .2)
        bottom = (height - avg_y) + (box_radius) - (box_radius * .2)
        right = avg_x + box_radius
        left = avg_x - box_radius

        image = image.crop((left, top, right, bottom))
        image = image.resize((512, 512))
        image.save(SAVE_IMAGES_DIR + filename.split('_')[-1])
        print()


def align_faces_eyes():
    # preds = fa.get_landmarks_from_directory(RAW_IMAGES_DIR)
    for filename in [f for f in os.listdir(RAW_IMAGES_DIR) if f[0] not in '._']:
        input = io.imread(RAW_IMAGES_DIR + filename)
        preds = fa.get_landmarks(input)
        image = Image.open(RAW_IMAGES_DIR + filename)

        box_radius = 225

        y = preds[0][:, 1][36:48]
        x = preds[0][:, 0][36:48]
        width, height = image.size
        avg_y = np.mean(y)
        avg_x = np.mean(x)

        top = (height - avg_y) - (box_radius) - (box_radius * .5)
        bottom = (height - avg_y) + (box_radius) - (box_radius * .5)
        right = avg_x + box_radius
        left = avg_x - box_radius

        image = image.crop((left, top, right, bottom))
        image = image.resize((512, 512))
        image.save(SAVE_IMAGES_DIR + filename.split('_')[-1])


if __name__ == "__main__":
    align_faces_eyes()
