ntu=60 st=r
ve=shift le=ViT-B/32
nc=10 nepc=1700 ls=100 mode=train gpu=0
th=50 t=2

python gen_text_feat.py --ntu $ntu --arch $le --gpu $gpu

##'''
for ss in 5 12
do
    tdir="sk_feats/shift_"$ss"_r/"
    edir="sk_feats/shift_val_"$ss"_r/"
    wdir_1="results/"$ss"_r"
    wdir_2="results/"$ss"_r_val"

    for tm in "lb" "ad" "md" "ad_md" "lb_ad_md"
    do
        echo "-----------------------------------"
        echo "NTU"$ntu"u"$ss" "$tm

        echo "=========="
        echo "Stage 1"
        echo "..."
        r1=`python train.py \
        --ntu $ntu --ss $ss --alpha 0.5 --lmd 100 --use_cr_fact 1 --version "neg_t" \
        --st $st --ve $ve --le $le --tm $tm --num_cycles $nc --num_epoch_per_cycle $nepc --latent_size $ls --gpu $gpu \
        --phase train --mode $mode --dataset $tdir --wdir $wdir_1`
        za=${r1:0-35:5} c=${r1:0-18:1}
        echo "Best ZSL Acc: "$za" on cycle "$c

        echo "=========="
        echo "Stage 2"
        echo "..."
        r2=`python train.py \
        --ntu $ntu --ss $ss --alpha 0.5 --lmd 100 --use_cr_fact 1 --version "neg_t" \
        --st $st --ve $ve --le $le --tm $tm --num_cycles $nc --num_epoch_per_cycle $nepc --latent_size $ls --gpu $gpu \
        --phase val --mode $mode --dataset $edir --wdir $wdir_2`

        echo "=========="
        echo "Stage 3"
        echo "..."
        r3=`python gating_train.py \
        --ntu $ntu --ss $ss --st $st --ve $ve --le $le --tm $tm --phase val --dataset $edir --wdir $wdir_2 --th $th --t $t`
        echo "thresh: "${r3:0-23:4}", temp: "${r3:0-1}

        echo "=========="
        echo "Stage 4"
        echo "..."
        r4=`python gating_eval.py \
        --ntu $ntu --ss $ss --st $st --phase train --dataset $tdir --wdir $wdir_1 --ve $ve --le $le --tm $tm \
        --thresh ${r3:0-23:4} --temp ${r3:0-1}`
        sa=${r4:15:5} ua=${r4:39:5} hm=${r4:0-6:5}
        echo "S_Acc: "$sa", U_Acc: "$ua", H_Mean: "$hm

        echo "=========="
        echo "Log"
        echo "..."
        python log.py --ntu $ntu --ss $ss --tm $tm --za $za --sa $sa --ua $ua --hm $hm --le $le --ls $ls --nepc $nepc
        echo "Finish"
    done
done
