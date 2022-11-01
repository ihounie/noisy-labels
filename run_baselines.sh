CUDA_VISIBLE_DEVICES=1
for seed in 0
do
    for noise in "clean" "worst" "aggre"
    do
        python3 main.py --dataset cifar10 --noise_type $noise --is_human --seed $seed --wandb_log
    done
done