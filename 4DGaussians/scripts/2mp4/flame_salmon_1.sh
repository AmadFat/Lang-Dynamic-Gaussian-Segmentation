export FPS=30
export RESW=480
export RESH=360

export INPUT_DIR=output/dynerf/flame_salmon_1/video/
export OUTPUT_DIR=~/cource/lang-segment-anything/videos/flame_salmon_1/
mkdir -p $OUTPUT_DIR

export ITER=5
ffmpeg -y -i $INPUT_DIR/ours_$ITER/renders/%05d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p -vf "scale=$RESW:$RESH" $OUTPUT_DIR/$ITER.mp4
export ITER=200
ffmpeg -y -i $INPUT_DIR/ours_$ITER/renders/%05d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p -vf "scale=$RESW:$RESH" $OUTPUT_DIR/$ITER.mp4
export ITER=1000
ffmpeg -y -i $INPUT_DIR/ours_$ITER/renders/%05d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p -vf "scale=$RESW:$RESH" $OUTPUT_DIR/$ITER.mp4
export ITER=5000
ffmpeg -y -i $INPUT_DIR/ours_$ITER/renders/%05d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p -vf "scale=$RESW:$RESH" $OUTPUT_DIR/$ITER.mp4
export ITER=20000
ffmpeg -y -i $INPUT_DIR/ours_$ITER/renders/%05d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p -vf "scale=$RESW:$RESH" $OUTPUT_DIR/$ITER.mp4