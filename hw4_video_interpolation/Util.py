import os
import imageio
from PIL import Image
from os.path import join
import time
import tqdm

def extract_frames (video_path, data_dir):
    print ("\n\n[*] Extract frames from {}".format (video_path))
    def convert_frame (x):
        return Image.fromarray (x[:, :, :3], mode="RGB")

    video_reader = imageio.get_reader (video_path, "ffmpeg")
    print ("[*] There are totally: {} frames".format (len (video_reader)))
    fps = video_reader.get_meta_data().get ("fps", None)

    #frames = [convert_frame (x) for x in video_reader]
    
    frames = []
    i = 0
    for i, img in enumerate (video_reader):
        try:
            frame = convert_frame (img)
            #frames.append (frame)

            file_path = join (data_dir, "{:05d}.png".format (i))

            frame.save (file_path)
        except:
            break

        if i == len (video_reader)-10:
            break

    print ("[*] There are: {} good frames".format (len (frames)))

    print ("[*] End of extracting.")

    return frames, fps

def save_frames (data_dir, frames):
    print ("[*] Save frames to {}".format (data_dir))
    s_time = time.time ()
    length = len (frames)
    for i in range (length):
        frame = frames [i]
        file_path = join (data_dir, "{:05d}.png".format (i))

        frame.save (file_path)
    e_time = time.time()

    print ("[*] Done. Takes {}s".format (e_time-s_time))

path = "data/train/video"

for i in range (4,5):
    video_path =  "{}{}.mp4".format (path, i+1)
    data_dir = "{}{}".format (path, i+1)

    if not os.path.exists (data_dir):
        os.makedirs (data_dir)

    frames, fps = extract_frames (video_path, data_dir)

    #save_frames (data_dir, frames)

    del frames

