import os

image_files = []
os.chdir('/content/drive/MyDrive/yolov4_kaggle/DATA/TRAIN/')
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append("/content/drive/MyDrive/yolov4_kaggle/DATA/TRAIN/" + filename)
os.chdir("..")
with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()