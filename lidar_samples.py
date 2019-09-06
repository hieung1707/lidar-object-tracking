import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MeanShift
from test import EllipsoidTool


curr_id = 0


def get_ranges(filename):
    samples = []
    with open(filename) as fp:
        lines = fp.readlines()
        for line in lines:
            parts = line.split(':')
            if parts[0] == 'ranges':
                samples_raw = parts[1]
                # cut [ ]
                samples_raw = samples_raw[1:-2]
                # replace inf to 0
                samples_raw = samples_raw.replace('inf', '100000')
                list_samples_str = samples_raw.split(',')[:-1]
                # parse to int list
                list_samples_float = list(map(float, list_samples_str))
                samples.append(list_samples_float)

    return np.array(samples)


def re_arrange(samples):
    x = []
    y = []
    for i, r in enumerate(samples):
        x.append(r*np.sin(2*np.pi*i/len(samples)))
        y.append(r*np.cos(2*np.pi*i/len(samples)))
    return x, y


def key_event(e):
    global curr_pos
    global plots
    global labels
    global dists
    global centroids

    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    else:
        return
    curr_pos = curr_pos % len(plots)
    print('Number of objects: {}'.format(len(all_clusters[curr_pos])))
    ax.cla()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    # ax.scatter(original_plots[curr_pos][0], original_plots[curr_pos][1], alpha=0.5, s=20)
    ax.scatter(plots[curr_pos][0], plots[curr_pos][1], c=labels[curr_pos], alpha=0.5, s=20)
    ax.scatter(centroids[curr_pos][0], centroids[curr_pos][1], c=cent_labels[curr_pos], marker='D', s=30)
    ax.scatter([0], [0], marker='D', s=20)
    fig.canvas.draw()


def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def check_pov(x, y):
    A = (0, 0)
    B1 = (np.sqrt(3) + 0.175, 1)
    B2 = (-np.sqrt(3) - 0.175, 1)
    # list_qualified = []
    list_indices = []
    for i in range(len(x)):
        x_temp, y_temp = x[i], y[i]
        d1 = (x_temp - A[0])*(B1[1] - A[1]) - (y_temp - A[1])*(B1[0] - A[0])
        d2 = (x_temp - A[0])*(B2[1] - A[1]) - (y_temp - A[1])*(B2[0] - A[0])
        if d1 <= 0 <= d2:
            # list_qualified.append((x_temp, y_temp))
            list_indices.append(i)
    return list_indices


def filter_n_merge_cluster(x, y, label, thresh=0, max_dist=0.3):
    x, y, label = filter_labels(x, y, label, thresh)
    diffs = np.diff(label)
    indices = np.where(diffs != 0)[0].tolist()
    current_diff_idx = 0
    idx_len = len(indices)
    clusters = []
    i = 0
    while current_diff_idx < idx_len:
        p1 = (x[indices[current_diff_idx]], y[indices[current_diff_idx]])
        p2 = (x[indices[current_diff_idx] + 1], y[indices[current_diff_idx] + 1])

        distance = calculate_distance(p1, p2)
        if distance <= max_dist:
            lower_boundry = indices[current_diff_idx] + 1
            upper_boundry = (indices[(current_diff_idx + 1) % idx_len] + 1) % len(label)
            merge_range = range(lower_boundry, upper_boundry) if lower_boundry <= upper_boundry else list(
                range(lower_boundry, len(label))) + list(range(upper_boundry))
            for j in merge_range:
                label[j] = label[(indices[current_diff_idx] - 1) % len(label)]
            indices.pop(current_diff_idx)
            idx_len -= 1
        else:
            current_diff_idx += 1
    diffs = np.diff(label)
    indices = np.where(diffs != 0)[0].tolist()
    for idx in range(len(indices)):
        start = (indices[idx] + 1) % len(label)
        end = (indices[(idx + 1) % len(indices)])
        cluster_range = list(range(start, end)) if start <= end else list(range(start, len(label))) + list(range(0, end))
        cluster = [(x[i], y[i]) for i in cluster_range]
        clusters.append(cluster)
    return x, y, label, clusters


def filter_labels(x, y, label, thresh=0):
    x = [x[i] for i in range(len(label)) if label[i] >= thresh]
    y = [y[i] for i in range(len(label)) if label[i] >= thresh]
    label = [l for l in label if l >= thresh]
    return x, y, label


def calculate_centroid(x, y, clusters):
    x_cent, y_cent = [], []
    print('start')
    for cluster in clusters:
        avg_x = 0
        avg_y = 0
        for idx in cluster:
            avg_x += x[idx]
            avg_y += y[idx]
        if len(cluster) >= 6:
            x_cent.append(avg_x / (len(cluster) + 1e-6))
            y_cent.append(avg_y / (len(cluster) + 1e-6))
    print('end')
    return x_cent, y_cent


def coefficient(p1, p2 ,p3):
    x_1 = p1[0]
    x_2 = p2[0]
    x_3 = p3[0]
    y_1 = p1[1]
    y_2 = p2[1]
    y_3 = p3[1]

    a = y_1/((x_1-x_2)*(x_1-x_3)) + y_2/((x_2-x_1)*(x_2-x_3)) + y_3/((x_3-x_1)*(x_3-x_2))

    b = (-y_1*(x_2+x_3)/((x_1-x_2)*(x_1-x_3))
         -y_2*(x_1+x_3)/((x_2-x_1)*(x_2-x_3))
         -y_3*(x_1+x_2)/((x_3-x_1)*(x_3-x_2)))

    c = (y_1*x_2*x_3/((x_1-x_2)*(x_1-x_3))
        +y_2*x_1*x_3/((x_2-x_1)*(x_2-x_3))
        +y_3*x_1*x_2/((x_3-x_1)*(x_3-x_2)))
    return a, b, c


if __name__ == '__main__':
    curr_pos = 0
    # refine samples
    samples_np = get_ranges('lidar_samples/1-1.txt')
    plots = []
    labels = []
    dists = []
    all_candidates = []
    centroids = []
    cent_labels = []
    original_plots = []
    all_clusters = []
    model = DBSCAN(eps=0.5, min_samples=6)
    ET = EllipsoidTool()
    for idx in range(samples_np.shape[0]):
        cent_label = []
        x, y = re_arrange(samples_np[idx])
        # original_plots.append((x, y))
        points = list(zip(x, y))
        model.fit(points)
        label = model.labels_
        thresh = 0
        # x, y, label = filter_labels(x, y, label, thresh=thresh)
        x, y, label, clusters = filter_n_merge_cluster(x, y, label, thresh=thresh, max_dist=0.2)
        center_x = []
        center_y = []
        for cluster in clusters:
            if cluster == []:
                continue
            cluster_np = np.asarray(cluster)
            (center, radii, rotation) = ET.getMinVolEllipse(cluster_np, .01)
            center_x.append(center[0])
            center_y.append(center[1])
        plots.append((x, y))
        labels.append(label)
        centroids.append((center_x, center_y))
        cent_labels.append(np.arange(len(center_x)))
        all_clusters.append(clusters)

    fig = plt.figure(figsize=(10, 10))
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax = fig.add_subplot(111)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    print('Number of objects: {}'.format(len(all_clusters[curr_pos])))
    # ax.scatter(original_plots[curr_pos][0], original_plots[curr_pos][1], alpha=0.5, s=20)
    ax.scatter(plots[curr_pos][0], plots[curr_pos][1], c=labels[curr_pos], alpha=0.5, s=20)
    ax.scatter(centroids[curr_pos][0], centroids[curr_pos][1], c=cent_labels[curr_pos], marker='D', s=30)
    ax.scatter([0], [0], marker='D', s=20)
    plt.show()
