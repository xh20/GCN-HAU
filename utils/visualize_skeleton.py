import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
import json
from PIL import Image
from matplotlib.patches import Rectangle


def plot_3d_skeleton(data, num_joint, edges, save_dir):
    # sk_data: 1 x C x T x V x 1
    data = data.squeeze()
    C, T, V = data.shape

    sk = data[:, :, :num_joint]
    objects = data[:, :, num_joint:]

    colormap = cm.get_cmap('viridis')
    colormap_o = cm.get_cmap('gist_heat')

    # Generate RGB values for 5 different points along the colormap
    colors = [colormap(i / (num_joint - 1)) for i in range(num_joint)]
    colors_o = [colormap_o(i / (V - num_joint - 1)) for i in range(V - num_joint)]

    # fig = plt.figure(figsize=(120, 10))
    # # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.set_aspect(0.1)  # Smaller aspect values stretch the x-axis more
    # ax.set_axis_off()
    # ax.grid(True)
    # Adjust the viewing angle: Front-right view
    ## Behave
    # ax.view_init(elev=90, azim=-90)
    # ax.view_init(elev=90, azim=90)
    # gap = 1
    ## Bimacts
    gap = 500
    size = 60

    for frame in range(T):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()
        ax.grid(False)
        ax.set_xlim([-600, 800])
        ax.set_ylim([-400, 600])
        ax.view_init(elev=-80, azim=-90)
        # distance = -frame*gap
        distance = 1
        x, y, z, c = sk[0, frame, :], sk[1, frame, :], sk[2, frame, :], sk[3, frame, :]
        for i, (x_coord, y_coord, z_coord) in enumerate(zip(x, y, z)):
            ax.scatter(x_coord + distance, y_coord, z_coord, color=colors[i], s=size)
        # Plot skeleton connections
        for start, end in edges:
            if c[start] > 0.1 and c[end] > 0.1:
                ax.plot(
                    [x[start] + distance, x[end] + distance],
                    [y[start], y[end]],
                    [z[start], z[end]],
                    color=colors[start],
                    linewidth=size // 20
                )
        x_o, y_o, z_o, c_o = objects[0, frame, :], objects[1, frame, :], objects[2, frame, :], objects[3, frame, :]
        for i, (x_coord, y_coord, z_coord, confidence) in enumerate(zip(x_o, y_o, z_o, c_o)):
            if confidence > 0.5:
                ax.scatter(x_coord + distance, y_coord, z_coord, color=colors_o[i], s=4 * size)
    # ax.view_init(elev=-90, azim=-90)
    # ax.set_box_aspect([1, 1, 1])
    #     plt.savefig(save_dir / f"3d_skeleton_3fps_{frame}.svg", format="svg", transparent=True)
    #     plt.close(fig)
    # Save as SVG
    plt.show()
    print("done")

def plot_3d_skeleton_in_figure(data, num_joint, edges, save_dir):
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    # sk_data: 1 x C x T x V x 1
    data = data.squeeze()
    C, T, V = data.shape

    sk = data[:, :, :num_joint]
    objects = data[:, :, num_joint:]

    colormap = cm.get_cmap('viridis')
    colormap_o = cm.get_cmap('gist_heat')

    # Generate RGB values for 5 different points along the colormap
    colors = [colormap(i / (num_joint - 1)) for i in range(num_joint)]
    colors_o = [colormap_o(i / (V - num_joint - 1)) for i in range(V - num_joint)]

    # fig = plt.figure(figsize=(120, 10))
    # # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.set_aspect(0.1)  # Smaller aspect values stretch the x-axis more
    # ax.set_axis_off()
    # ax.grid(True)
    # Adjust the viewing angle: Front-right view
    ## Behave
    # ax.view_init(elev=90, azim=-90)
    # ax.view_init(elev=90, azim=90)
    # gap = 1
    ## Bimacts
    gap = 500
    size = 60
    step = 1

    for frame in range(0, T, step):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()
        ax.grid(False)
        ax.set_xlim([-600, 800])
        ax.set_ylim([-500, 600])
        ax.view_init(elev=-80, azim=-90)
        # distance = -frame*gap
        distance = 0
        x, y, z, c = sk[0, frame, :], sk[1, frame, :], sk[2, frame, :], sk[3, frame, :]
        for i, (x_coord, y_coord, z_coord) in enumerate(zip(x, y, z)):
            ax.scatter(x_coord + distance, y_coord, z_coord, color=colors[i], s=size)
        # Plot skeleton connections
        for start, end in edges:
            if c[start] > 0.1 and c[end] > 0.1:
                ax.plot(
                    [x[start] + distance, x[end] + distance],
                    [y[start], y[end]],
                    [z[start], z[end]],
                    color=colors[start],
                    linewidth=size // 20
                )
        x_o, y_o, z_o, c_o = objects[0, frame, :], objects[1, frame, :], objects[2, frame, :], objects[3, frame, :]
        for i, (x_coord, y_coord, z_coord, confidence) in enumerate(zip(x_o, y_o, z_o, c_o)):
            if confidence > 0.5:
                ax.scatter(x_coord + distance, y_coord, z_coord, color=colors_o[i], s=4 * size)
        plt.savefig(save_dir / f"3d_skeleton_3fps_{frame}.svg", format="svg", transparent=True)
        plt.close(fig)
    # ax.view_init(elev=-90, azim=-90)
    # ax.set_box_aspect([1, 1, 1])

    # Save as SVG
    # plt.show()
    print("done")


def read_data(data_path, l_filename, r_filename, save_dir):
    with open(data_path / l_filename) as f:
        l_data = json.load(f)
    with open(data_path / r_filename) as f:
        r_data = json.load(f)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    num_joint = 12
    start = 0
    window_size = 240
    step = 10
    half_step = step // 2
    # T x V x C => C T V
    l_nodes = np.array(l_data["nodes"])[start:window_size:step, 2:5, :].transpose(2, 0, 1)
    r_nodes = np.array(r_data["nodes"])[start:window_size:step, :num_joint, :].transpose(2, 0, 1)
    objects = np.array(r_data["nodes"])[start:window_size:step, num_joint:, :].transpose(2, 0, 1)
    l_nodes[0, :, :] = -l_nodes[0, :, :]
    nodes = np.concatenate([r_nodes, l_nodes, objects], axis=-1)
    # nodes[0, :, :] = -nodes[0, :, :]
    path = r_data["color_path"][start + half_step:window_size:step]

    # images = []
    # for i, image_path in enumerate(path):
    #     img = Image.open(image_path)
    #     images.append(np.array(img))
    #     img.save(save_dir / f"img_{i}.png")
    # # Images: C x T x H x W
    # images = np.stack(images, axis=0)
    return nodes


if __name__ == '__main__':
    s = 1
    ts = 4
    tk = 7
    data_path = Path(f"/media/hao/data_base/BimanualActions/generated_data/subject_{s}/")
    l_filename = f"s{s}ts{ts}tk{tk}l.json"
    r_filename = f"s{s}ts{ts}tk{tk}r.json"
    save_dir1 = Path(f"/media/hao/data_base/Paper/IROS_boey/figure/s{s}ts{ts}tk{tk}/")
    nodes1 = read_data(data_path, l_filename, r_filename, save_dir1)

    s = 1
    ts = 7
    tk = 4
    data_path = Path(f"/media/hao/data_base/BimanualActions/generated_data/subject_{s}/")
    l_filename = f"s{s}ts{ts}tk{tk}l.json"
    r_filename = f"s{s}ts{ts}tk{tk}r.json"
    save_dir2 = Path(f"/media/hao/data_base/Paper/IROS_boey/figure/s{s}ts{ts}tk{tk}/")
    nodes2 = read_data(data_path, l_filename, r_filename, save_dir2)

    # save_dir3 = Path(f"/media/hao/data_base/Paper/IROS25/figure/mixup/")
    ## Behave dataset
    # num_joint = 22
    # edges = [(0, 1), (0, 2), (0, 3), (1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11), (3, 6), (6, 9), (9, 12),
    #          (12, 15), (9, 13), (13, 16), (16, 18), (18, 20), (9, 14), (14, 17), (17, 19), (19, 21)]
    ## Bimacs:
    edges = [(4, 3), (3, 2), (2, 1), (0, 1), (5, 1), (6, 5), (7, 5), (10, 8), (8, 0), (11, 9), (9, 0),
             (14, 13), (13, 12), (12, 1)]
    # edges = [(14, 13), (13, 12), (12, 1)]
    num_joint = 12

    # T x V x C => C T V

    plot_3d_skeleton_in_figure(nodes1, num_joint + 3, edges, save_dir1)
    plot_3d_skeleton_in_figure(nodes2, num_joint + 3, edges, save_dir2)
    # nodes_3 = 0.8*nodes1 + 0.2*nodes2
    # plot_3d_skeleton_in_figure(nodes_3, num_joint + 3, edges, save_dir3)

    # print(d[""].shape)
    # print(d["color"].shape)
