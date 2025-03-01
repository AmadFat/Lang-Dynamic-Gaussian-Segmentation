export RGB_DIR=videos/flame_salmon_1/
export SEG_DIR=output/dynerf/flame_salmon_1/video/
export OUTPUT_DIR=videos/flame_salmon_1/
mkdir -p $OUTPUT_DIR

export ITER=5
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER=200
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER=1000
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER=5000
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER=20000
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif