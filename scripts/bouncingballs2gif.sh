export RGB_DIR=4DGaussians/output/dnerf/bouncingballs/video/
export SEG_DIR=lang-segment-anything/output/dnerf/bouncingballs/
export RESW=

export ITER=5
ffmpeg -y -i $RGB_DIR/ours_$ITER/video_rgb.mp4 