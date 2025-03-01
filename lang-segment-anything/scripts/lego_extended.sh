export TT=0.25
export BT=0.05
export B=4
export SEED=25
export EXP="dnerf/lego_extended"
export RESW=360
export RESH=360

export ITER="45000"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="5"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="45000_0"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="45000_+30"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="15000"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="15000_0"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="15000_+30"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="5000"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="5000_0"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="5000_+30"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="1000"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="1000_0"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="1000_+30"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="200"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="200_0"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="200_+30"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="60"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER="20"
python seg.py -m large -t base wheel excavator -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH