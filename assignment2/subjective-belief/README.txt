Subjective Belief Experiment

-------------
Related paper
-------------

X. Zhu, T. Gedeon, S. Caldwell, R. Jones and X. Gu, "Deceit Detection: Identification of Presenter’s Subjective Doubt Using Affective Observation Neural Network Analysis," 2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Toronto, ON, Canada, 2020, pp. 3174-3181, doi: 10.1109/SMC42975.2020.9283210.

------------
Introduction
------------

In this experiment we aim to examine whether physiological signals from observers can be used to detect presenters' subjective belief in some information they present.

We first invited some presenters to present some information and for half of the cases, we told the presenters that the following content they were going to present were bogus but they had to present it naturally. In this way we were trying to manipulate their own belief in the content they were presenting. These presentations were recorded as videos.

After that we recruited some observers to watch these video presentations, and we asked observers to estimate whether the presenters' belief has been manipulated or not. While observers watched the video presentations we recorded their Blood Volume Pulse (BVP), Galvanic Skin Response (GSR), Skin Temperature (ST) and Pupillary Dilation (PD).

The goal was to build a classification model from these physiological signals to estimate the manipulation of presenters' belief, and whether the model is more accurate than observers' conscious verbal judgement.

------------
Dataset
------------

The "subjective_belief_observers_features_labels.csv" contains the physiological features of observers and respective labels. There were 123 columns and 368 rows. Each row represents the set of features extracted from participants' physiological signals when they watched one specific video.

More specifically this table is structured as follows.
    - The first column [pid_vid] contains a string that records the participant id and the id of the video this participant watched. Its format is "pid_vid"
    - The following 34 columns [0_bvp to 33_bvp] contains 34 features extracted from the participant's BVP when they watched a video
    - The following 23 columns [0_gsr to 22_gsr] contains 23 features extracted from the participant's GSR when they watched a video
    - The following 23 columns [0_temp to 22_temp] contains 23 features extracted from the participant's ST when they watched a video
    - The following 39 columns [0_eye to 38_eye] contains 39 features extracted from the participant's PD when they watched a video
    - The last column contains the label of whether the presenter in this video has doubted about their belief in the video. 1 represents we didn't manipulate presenters' subjective belief, and 0 means they doubted about what they presented as we manipulated their belief.

--------------------
Exclude participants
--------------------

1. Excluded p02, 04, 10, 19, 20, 35, 29 due to eye tracking failure
2. Excluded p05, p36 due to miss alignment of timestamps on E4 data and eye data
3. Excluded p33 due to the lack of eye data and e4 data
