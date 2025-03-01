export TT=0.15
export BT=0.1
export B=1
export SEED=21
export EXP="dynerf/cut_roasted_beef"
export RESW=480
export RESH=360

export ITER=20000
python seg.py -m large -t man dog curtain wall knife bottle cupboard beef table window -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER=5
python seg.py -m large -t man dog curtain wall knife bottle cupboard beef table window -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER=1000
python seg.py -m large -t man dog curtain wall knife bottle cupboard beef table window -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER=5000
python seg.py -m large -t man dog curtain wall knife bottle cupboard beef table window -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER=200
python seg.py -m large -t man dog curtain wall knife bottle cupboard beef table window -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH