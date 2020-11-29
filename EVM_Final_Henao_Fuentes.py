import cv2
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
import os
import time

# Functions used in Code

#Build Gaussian Pyramid using cv2 pyr methods
def build_gaussian_pyramid(src, level):
    s = src.copy()
    pyramid = [s]
    for i in range(level):
        s = cv2.pyrDown(s)
        pyramid.append(s)
    return pyramid

#load video from file and get usefull data from it
def load_video(video_filename):
    cap = cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # video frame count
    # frame width and height
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) # video fps
    # pre locate a numpy array so we can make a copy of our video
    video_tensor = np.zeros((frame_count,height,width,3),dtype='float')
    x=0
    # Make a copy of every frame of the video.
    while cap.isOpened():
        ret,frame = cap.read()
        if ret is True:
            video_tensor[x] = frame
            x = x + 1
        else:
            break
    return video_tensor,fps

# apply temporal ideal bandpass filter to gaussian video
# this is not a really good method to implement a bandpass filter though.
# such brutal changes in frequency will introduce distortion
def temporal_ideal_filter(tensor, low, high, fps, axis=0):
    fft=fftpack.fft(tensor,axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff=fftpack.ifft(fft, axis=axis)
    return np.abs(iff)

# build gaussian pyramid for video
def gaussian_video(video_tensor, levels):
    for i in range(0,video_tensor.shape[0]):
        frame = video_tensor[i]
        pyr = build_gaussian_pyramid(frame, levels)
        gaussian_frame = pyr[-1] #save the highest level of the pyr for every frame
        if i==0:
            #pre locate a variable to save the gaussian video
            vid_data = np.zeros((video_tensor.shape[0],gaussian_frame.shape[0],gaussian_frame.shape[1],3))
        vid_data[i] = gaussian_frame
    return vid_data

#amplify the video
def amplify_video(gaussian_vid,amplification):
    return gaussian_vid*amplification

#reconstruct video from original video and gaussian video
def reconstruct_video(amp_video, origin_video, levels):
    # pre - locate variable to reconstruct video
    final_video = np.zeros(origin_video.shape)
    for i in range(0,amp_video.shape[0]): #for every frame
        img = amp_video[i]
        for x in range(levels):# go up in gauss pyr
            img = cv2.pyrUp(img)
        img = img + origin_video[i] # reconstruct frame
        final_video[i] = img # save frame
    return final_video

#save video to files
def save_video(video_tensor):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    [height, width] = video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter("Color_AMP_prueba.avi", fourcc, 30, (width, height), 1)
    for i in range(0,video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()

if __name__=="__main__":
    path = 'C:/Users/ACER/Desktop/Semestre10/Imagenes/Proyecto'
    image_name = 'Murphy.mp4'
    #image_name = 'Juan_Murphy.mp4'
    #image_name = 'Mary.mp4'
    path_file = os.path.join(path, image_name)

    ###################################################################
    ####### Parameters that show good results for skin color amp ######
    ###################################################################
    levels = 3 #Number of gaussian pyramid levels constructed for every frame
    amplification = 10
    low_f = 0.8
    high_f = 1
    ###################################################################
    ############## Apply Color EVM and measure times ##################
    ###################################################################

    start = time.perf_counter()  # Measuring load video time in s
    t, f = load_video(path_file)
    stop = time.perf_counter()  # Measuring filter implementation time in s
    load_video_time = stop - start

    start = time.perf_counter()  # Measuring gaussian video time in s
    gaussVid = gaussian_video(t, levels)
    stop = time.perf_counter()  # Measuring gaussian time in s
    gauss_video_time = stop - start

    start = time.perf_counter()  # Measuring filter implementation time in s
    filtered_video = temporal_ideal_filter(gaussVid, low_f, high_f, f)
    stop = time.perf_counter()  # Measuring filter implementation time in s
    filter_time = stop - start

    amplified_video = amplify_video(filtered_video, amplification)

    start = time.perf_counter()  # Measuring reconstruction time in s
    final_video = reconstruct_video(amplified_video, t, levels)
    stop = time.perf_counter()  # Measuring reconstruction time in s
    video_time = stop - start

    save_video(final_video)
    print('loading video time is: ', load_video_time)
    print('gaussian video time construction is: ', gauss_video_time)
    print('filter implementation time is: ', filter_time)
    print('video reconstruction time is: ', video_time)



