export FPS=30
export EXP=output/dynerf/flame_salmon_1
mkdir -p $EXP/video

export ITER=5
ffmpeg -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER=200
ffmpeg -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER=1000
ffmpeg -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER=5000
ffmpeg -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER=20000
ffmpeg -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4