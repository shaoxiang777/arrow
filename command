
aist_entrance_hall_1
aist_store_2
arrow360/VIDEO_0665

xhost +local:


docker run -it --rm --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix:ro \
    -v ~/Documents/project/stella_vslam:/work \
    stella_vslam-iridescence

docker run -it --rm --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix:ro \
    -v ~/Documents/project/stella_vslam:/work \
    --gpus all \
    stella_vslam-desktop


/stella_vslam_examples/build/run_video_slam \
    -v /work/orb_vocab.fbow \
    -c /work/data/arrow360/VIDEO_0665/config.yaml \
    -m /work/data/arrow360/VIDEO_0665/video.mp4 \
    --frame-skip 2 \
    --no-sleep \
    --eval-log-dir /work/data/arrow360/VIDEO_0665/ \
    --map-db-out /work/data/arrow360/VIDEO_0665/my_saved_map.msg

evo_traj tum data/arrow360/VIDEO_0665/frame_trajectory.txt --ref data/arrow360/VIDEO_0665/keyframe_trajectory.txt -p --plot_mode=xyz

/stella_vslam_examples/build/run_video_slam \
    --disable-mapping \
    -v /work/orb_vocab.fbow \
    -c /work/data/arrow360/VIDEO_0665/config.yaml \
    -m /work/data/arrow360/VIDEO_0665/video.mp4 \
    --frame-skip 2 \
    --map-db-in /work/data/arrow360/VIDEO_0665/my_saved_map.msg