import os
import numpy as np
import open3d as o3d

if __name__ == '__main__':
    raw_data = np.load('/home/rvsa/gary318/reart/data/mbs-sapien/data/000212 copy.npz', allow_pickle=True)
    data_folder = "/home/rvsa/gary318/reart/data/raw_fridge_data/122601"

    # Read point cloud and downsample
    N_POINTS = 512
    pcds = []

    segm = np.ones((N_POINTS, ), np.int32)
    for i in range(3):
        pcd = np.loadtxt(os.path.join(data_folder, "state_%d.xyz" % i))
        pcd = pcd[np.random.choice(range(pcd.shape[0]), size=N_POINTS)]
        pcds.append(pcd)

        if i == 0:
            segm[(pcd[:, 2] > 0.587) & (pcd[:, 1] < -0.076)] = 2
            segm[(pcd[:, 2] > 0.587) & (pcd[:, 1] >= -0.076)] = 3

    pc = np.stack(pcds + [pcds[-1]], axis=0)
    pc = pc - np.mean(pc, axis=(0, 1), keepdims=True)
    pc /= (np.std(pc) / 0.1)

    # np.random.randint(0, 3, size=(4, N_POINTS))

    segm = np.concatenate([segm.reshape((1, N_POINTS)), raw_data["segm"][1:]])

    np.savez('/home/rvsa/gary318/reart/data/mbs-sapien/data/000212.npz', segm=segm, pc=pc, trans=raw_data["trans"])
