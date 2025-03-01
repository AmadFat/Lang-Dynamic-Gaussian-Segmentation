export TT=0.1
export BT=0.075
export B=1
export SEED=60
export EXP="dynerf/flame_salmon_1"
export RESW=480
export RESH=360

export ITER=20000
python seg.py -m large -t man meat curtain wall flame bottle cupboard table window cup house wok knife -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER=5
python seg.py -m large -t man meat curtain wall flame bottle cupboard table window cup house wok knife -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER=1000
python seg.py -m large -t man meat curtain wall flame bottle cupboard table window cup house wok knife -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER=5000
python seg.py -m large -t man meat curtain wall flame bottle cupboard table window cup house wok knife -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH
export ITER=200
python seg.py -m large -t man meat curtain wall flame bottle cupboard table window cup house wok knife -tt $TT -bt $BT -i input/$EXP/video/ours_$ITER/renders/ -o output/$EXP/$ITER/ -b $B -s $SEED -rw $RESW -rh $RESH