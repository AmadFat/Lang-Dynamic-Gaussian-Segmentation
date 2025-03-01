export FPS=30
export EXP=output/dnerf/lego_extended
mkdir -p $EXP/video

export ITER="1000"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="1000_+30"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="1000_0"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="5000"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="5000_+30"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="5000_0"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="15000"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="15000_+30"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="15000_0"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="45000"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="45000_+30"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="45000_0"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="5"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="20"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="60"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="200"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="200_+30"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4
export ITER="200_0"
ffmpeg -y -i $EXP/$ITER/%5d.png -framerate $FPS -c:v libx264 -pix_fmt yuv420p $EXP/video/$ITER.mp4