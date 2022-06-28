import cv2
from sort import *


def filter_classes(classes_in, scores_in, boxes_in):
    classes_out = np.array([], dtype=int)
    scores_out = np.array([])
    boxes_out = np.empty(shape=[0, 4], dtype=int)
    for (class_id, score, box) in zip(classes_in, scores_in, boxes_in):
        if class_id in (1, 2, 3, 5, 6, 7, 8):
            classes_out = np.append(classes_out, class_id)
            scores_out = np.append(scores_out, score)
            boxes_out = np.append(boxes_out, [box], axis=0)
    return classes_out, scores_out, boxes_out


def init_YOLO(path_weigths,
              path_config,
              size=(416, 416)):
    n_net = cv2.dnn.readNet(path_weigths, path_config)
    n_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    n_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    model = cv2.dnn_DetectionModel(n_net)
    model.setInputParams(size=size, scale=1 / 255, swapRB=True)

    return model


def main():
    net_model = init_YOLO('configs/yolov4.weights',
                          'configs/yolov4.cfg')

    CONFIDENCE_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.4
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

    # class_names = []
    # with open("configs/classes.txt", "r") as f:
    #     class_names = [cname.strip() for cname in f.readlines()]

    cap = cv2.VideoCapture("example1.h264")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'fps: {fps}')

    mot_tracker = Sort(max_age=12, min_hits=2, iou_threshold=0.4)

    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            cv2.moveWindow("CSI Camera", 0, 0)
            # print(f'Frame:{frame_i}')

            grabbed, img_i = cap.read()
            if not grabbed:
                print('Exit from the video')
                exit()

            classes, scores, boxes = net_model.detect(img_i,
                                                      CONFIDENCE_THRESHOLD,
                                                      NMS_THRESHOLD)

            # Take into account only cars, moto, trucks etc
            filt_classes, filt_scores, filt_boxes = filter_classes(classes,
                                                                   scores,
                                                                   boxes)
            # # _______________________SORT_______________________
            # Convert bboxes from (startX, startY, H,W)
            # to (startX, startY,stopX,stopY)
            try:
                detections_SORT = filt_boxes.copy()
                detections_SORT[:, 2:4] += detections_SORT[:, 0:2]
            except BaseException:
                detections_SORT = np.empty((0, 5))
            tracked_objects = mot_tracker.update(detections_SORT)

            # #___________________________________________________

            for x1_id, y1_id, x2_id, y2_id, obj_id in tracked_objects:
                obj_id = int(obj_id)
                cv2.rectangle(img_i,
                              (int(x1_id), int(y1_id)),
                              (int(x2_id), int(y2_id)),
                              (0, 255, 0), 4)
                label = "ID: %d" % obj_id
                cv2.putText(img_i,
                            label,
                            (int(x1_id),
                             int(y1_id) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 255, 0),
                            4)

            scale_size = 80  # window size 16:9
            img_i = cv2.resize(img_i, (16 * scale_size, 9 * scale_size))
            cv2.imshow("CSI Camera", img_i)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 32:  # press white space
                break
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
