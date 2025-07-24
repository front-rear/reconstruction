import numpy as np
import pickle

if __name__ == '__main__':
    # peep_npz = np.load('/home/rvsa/gary318/reart/exp/sapien_212/result.pkl', allow_pickle=True)
    peep_npz = np.load('/home/rvsa/gary318/reart/data/mbs-sapien/data/000212 copy.npz', allow_pickle=True)
    pc = peep_npz["pc"].copy()
    pc[2] = pc[1].copy()
    pc[3] = pc[1].copy()

    np.savez('/home/rvsa/gary318/reart/data/mbs-sapien/data/000212.npz', segm=peep_npz["segm"], pc=pc, trans=peep_npz["trans"])
    print()