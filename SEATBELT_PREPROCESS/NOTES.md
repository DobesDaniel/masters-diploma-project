1. rotate/flip all videos i directory
    > rotate_video.py / flip_video.py
2. test skeleton and landmark detection
    > detect_skeleton.py
3. annotate actions in the videos
    > annotate_video.py
4. cut action clips from the videos
    > cut_clips.py
5. used for feature calculation ( + basic geometry)
    > calculate_features.py
6. extract landmarks for each action
   > extract_landmarks_from_clips.py
   
7.  create training data from landmarks:
    - different counts of frames per action detection
    - different offsets between frames
    - action frame window = 120 frames (2 seconds)
    
    - options:
        - 3 frames, offset 40 frames
          - 0,40,80
          - 40,80,120
        - 4 frames, offset 30 frames
          - 0,30,60,90
          - 30,60,90,120
        - 5 frames, offset 24 frames
          - 0,24,48,72,96
          - 24,48,72,96,120
        - 6 frames, offset 20 frames
          - 0,20,40,60,80,100
          - 20,40,60,80,100,120
    
    - d...distance | S...area | a...angle
    - features for training: 
        - A) Distances between points (4):
            d(NOSE, LEFT_ELBOW), d(NOSE, RIGHT_ELBOW), d(LEFT_SHOULDER, LEFT_WRIST), d(RIGHT_SHOULDER, RIGHT_WRIST)
            - normalized by d(LEFT_SHOULDER, RIGHT_SHOULDER)
        - B) Distances between points (4):
            d(NOSE, LEFT_ELBOW), d(NOSE, RIGHT_ELBOW), d(NOSE, LEFT_WRIST), d(NOSE, RIGHT_WRIST)
            - normalized by d(LEFT_SHOULDER, RIGHT_SHOULDER)
        - C) Area between points (4):
            S(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), S(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
            S(NOSE, LEFT_ELBOW, RIGHT_ELBOW), S(NOSE, LEFT_WRIST, RIGHT_WRIST)
            - normalized by d(LEFT_SHOULDER, RIGHT_SHOULDER)
        - D) Distances + Area between points (4):
            S(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), S(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            d(NOSE, LEFT_WRIST), d(NOSE, RIGHT_WRIST)
            - normalized by d(LEFT_SHOULDER, RIGHT_SHOULDER)
        - E) Angles between points (4):
            a(NOSE, LEFT_SHOULDER, LEFT_ELBOW), a(NOSE, RIGHT_SHOULDER, RIGHT_ELBOW),
            a(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), a(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    > create_dataset.py

8. Split experiments data into Train and Validation sets - split ratio: 0.8
    > split_data.py

---------------------------------------------------

9. Combine 2 directories of splitted data into 1
    > combine_experiments_splits.py

10. Annotations 2 - creating intervals of Ground Truth for test video
    > create_csv_annotations_for_test_video.py