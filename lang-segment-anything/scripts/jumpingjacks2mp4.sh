export FPS=25
export EXP=output/dnerf/jumpingjacks
mkdir -p $EXP/video

export ITER=5
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER=50
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER=500
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER=2000
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER=20000
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4