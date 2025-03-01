export FPS=25
export RESW=360
export RESH=360

export INPUT_DIR=output/dnerf/lego/video/
export OUTPUT_DIR=~/cource/lang-segment-anything/videos/lego/
mkdir -p $OUTPUT_DIR

export ITER=5
ffmpeg -y -i $INPUT_DIR/ours_$ITER/renders/%05d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p -vf "scale=$RESW:$RESH" $OUTPUT_DIR/$ITER.mp4
export ITER=50
ffmpeg -y -i $INPUT_DIR/ours_$ITER/renders/%05d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p -vf "scale=$RESW:$RESH" $OUTPUT_DIR/$ITER.mp4
export ITER=500
ffmpeg -y -i $INPUT_DIR/ours_$ITER/renders/%05d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p -vf "scale=$RESW:$RESH" $OUTPUT_DIR/$ITER.mp4
export ITER=2000
ffmpeg -y -i $INPUT_DIR/ours_$ITER/renders/%05d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p -vf "scale=$RESW:$RESH" $OUTPUT_DIR/$ITER.mp4
export ITER=20000
ffmpeg -y -i $INPUT_DIR/ours_$ITER/renders/%05d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p -vf "scale=$RESW:$RESH" $OUTPUT_DIR/$ITER.mp4