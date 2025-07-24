import os

path = "/home/rvsa/gary318/3D_Object_Reconstruction/dataset/121101/mask"
imgs = os.listdir(path)
for img_name in imgs:
    if img_name.endswith("_mask.png"):
        os.rename(os.path.join(path, img_name), os.path.join(path, img_name.replace("_mask.png", ".png")))