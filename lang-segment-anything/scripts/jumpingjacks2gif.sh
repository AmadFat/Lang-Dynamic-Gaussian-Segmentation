export RGB_DIR=videos/jumpingjacks/
export SEG_DIR=output/dnerf/jumpingjacks/video/
export OUTPUT_DIR=videos/jumpingjacks/
mkdir -p $OUTPUT_DIR

export ITER=5
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER=50
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER=500
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER=2000
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER=20000
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif