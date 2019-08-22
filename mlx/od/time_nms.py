import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlx.od.nms import compute_nms

def time_nms():
    all_num_boxes = np.arange(20, 1000, 100)
    num_labels = 20
    num_trials = 2

    times = []
    for num_boxes in all_num_boxes:
        start_time = time.process_time()

        for i in range(num_trials):
            yx = np.random.randint(0, 50, (num_boxes, 2))
            hw = np.random.randint(10, 30, (num_boxes, 2))
            boxes = np.concatenate((yx, yx+hw), axis=1)
            labels = np.random.randint(0, num_labels, (num_boxes,))
            scores = np.random.uniform(size=(num_boxes,))
            good_inds = compute_nms(boxes, labels, scores, iou_thresh=0.5)
            # print('prob of good boxes: ', len(good_inds) / num_boxes)

        elapsed_time = time.process_time() - start_time
        time_per_trial = elapsed_time / num_trials
        times.append(time_per_trial)
        # print('num_boxes: ', num_boxes)
        # print('time per nms: ', elapsed_time / num_trials)
    plt.plot(all_num_boxes, np.array(times))
    plt.xlabel('num boxes')
    plt.ylabel('time per nms')
    plt.savefig('/opt/data/plot.png')

if __name__ == '__main__':
    time_nms()