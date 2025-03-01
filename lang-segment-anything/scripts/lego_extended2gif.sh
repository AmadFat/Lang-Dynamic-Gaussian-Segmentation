export RGB_DIR=videos/lego_extended/
export SEG_DIR=output/dnerf/lego_extended/video/
export OUTPUT_DIR=videos/lego_extended/
mkdir -p $OUTPUT_DIR

export ITER="45000"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="5"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="45000_0"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="45000_+30"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="15000"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="15000_0"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="15000_+30"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="5000"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="5000_0"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="5000_+30"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="1000"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="1000_0"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="1000_+30"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="200"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="200_0"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="200_+30"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="60"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif
export ITER="20"
ffmpeg -y -i $RGB_DIR/$ITER.mp4 -i $SEG_DIR/$ITER.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 0:a? $OUTPUT_DIR/$ITER.gif