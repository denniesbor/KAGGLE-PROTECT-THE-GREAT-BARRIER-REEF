import os

image_files = []
os.chdir('/content/drive/MyDrive/yolov4_kaggle/DATA/TEST')
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append("/content/drive/MyDrive/yolov4_kaggle/DATA/TEST/" + filename)
os.chdir("..")
with open("test.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()