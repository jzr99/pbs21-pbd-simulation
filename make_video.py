import cv2
import os
import tqdm

fps = 30
image_folder = 'images/1639180459'
name = 'video_{:3d}.mp4'.format(fps)
video_path = os.path.join(image_folder, name)

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

writer = cv2.VideoWriter()
video = cv2.VideoWriter(filename=video_path, fourcc=writer.fourcc('m', 'p', '4', 'v'), fps=30, frameSize=(width, height))

for image in tqdm.tqdm(images):
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()