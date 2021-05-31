import os
import numpy as np
import cv2 as cv
import time
from PIL import Image




BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
               "Background": 15}

POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                   ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                   ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

inWidth = 368
inHeight = 368
thr = 0.1

proto = 'pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt'
model = 'pose/mpi/pose_iter_160000.caffemodel'

net = cv.dnn.readNetFromCaffe(proto, model)
image_dir = 'bikini_data'
success = 0
falt = 0
for input in os.listdir(image_dir):

    frame = cv.imread(image_dir + '/' + input)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inp = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                  (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    start_t = time.time()
    out = net.forward()

    kwinName = "Pose Estimation Demo: Cv-Tricks.com"
    cv.namedWindow(kwinName, cv.WINDOW_AUTOSIZE)
    try:

        points = []
        for i in range(len(BODY_PARTS)):

            heatMap = out[0, i, :, :]

            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > thr else None)

        y_aspect_ratio = points[8][1] - points[0][1]
        x_aspect_ratio = y_aspect_ratio

        top = [points[0][0], points[0][1]]
        bottom = [points[0][0], points[0][1] + y_aspect_ratio]
        left = [(points[0][0] - (x_aspect_ratio/2)), (points[0][1] + (y_aspect_ratio/2))]
        right = [(points[0][0] + (x_aspect_ratio/2)), (points[0][1] + (y_aspect_ratio/2))]

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        im = Image.fromarray(np.uint8(frame))
        im = im.crop((left[0], top[1], right[0], bottom[1]))
        # im.save('output/{}'.format(input))
        success += 1
        # print('sucess')


    except Exception as e:
        print(e)
        falt += 1

            # print("Success: {}".format(success))
            # print("fault: {}".format(falt))

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (255, 74, 0), 3)
            cv.ellipse(frame, points[idFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
            cv.putText(frame, str(idFrom), points[idFrom], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv.LINE_AA)
            cv.putText(frame, str(idTo), points[idTo], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv.LINE_AA)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv.LINE_AA)

    cv.imshow(kwinName, frame)
    cv.imwrite('output/'+input,frame)