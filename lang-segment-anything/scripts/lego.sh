export TT=0.275
export BT=0.05
export B=4
export SEED=29
export EXP="dnerf/lego"
export RESW=360
export RESH=360

export ITER=20000
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER=50
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER=500
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER=2000
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER=5
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH