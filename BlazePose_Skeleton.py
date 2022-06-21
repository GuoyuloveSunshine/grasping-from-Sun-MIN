import cv2
import mediapipe as mp
import os
import numpy as np
import json
mp_pose = mp.solutions.pose

#Process all Video, note the position x,y,z
total_path = "../data"

for group_name in os.listdir(total_path):
    group_path = os.path.join(total_path, group_name)
    video_path = os.path.join(group_path,"videosOriginal")
    blazepose_path = os.path.join(group_path,"blazepose")
    # print(video_path)

    if not os.path.exists(blazepose_path):
        os.mkdir(blazepose_path)
    video_dir = os.listdir(video_path)
    video_dir = sorted(video_dir)
    for video in video_dir:
        file_path_read = os.path.join(video_path,video)
        file_path_write = os.path.join(blazepose_path,video.split('.')[0]+'.json')
        frame_len = 1.0
        dic = {'positions':{}}
        print(file_path_read)
        print(file_path_write)
        # file_to_write=open(file_path_write,"w+")
        cap = cv2.VideoCapture(file_path_read)
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break
                # Flip the image horizontally for a later selfie-view display, and convert

            # the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
                image.flags.writeable = False
                results = pose.process(image)

                if not results.pose_landmarks:
                    continue
                frame = {}
                for keypoint in mp_pose.PoseLandmark:
                    name = str(keypoint).split(".")[-1]
                    name = name.capitalize()
        #             print(name)
                    frame[name] = [results.pose_landmarks.landmark[keypoint].x,
                                    results.pose_landmarks.landmark[keypoint].y,
                                  results.pose_landmarks.landmark[keypoint].z]

        #         print(frame_len, frame)
                dic["positions"][str(frame_len)] = frame
                frame_len += 1
            # print(dic)
            with open(file_path_write, 'w') as json_file:
                json.dump(dic, json_file)


            cap.release()
            
            json_file.close()
            print(len(dic["positions"]))