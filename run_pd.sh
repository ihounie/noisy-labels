CUDA_VISIBLE_DEVICES=1
for seed in 0
do
    for noise in "aggre"
    do
            for delta in 100.0
            do
                for huber_a in 1.0
                do
                for epsilon in 0.15
                    do
                    python3 main_pd.py --dual_lr 0.1 --dual_momentum 0.9 --perturbation_lr 0.01 --dataset cifar10 --noise_type $noise --is_human --seed $seed --wandb_log --huber_a $huber_a --huber_delta $delta --epsilon $epsilon
                    done
            done
        done
    done
done