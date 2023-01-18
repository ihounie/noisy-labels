CUDA_VISIBLE_DEVICES=1
for seed in 0 1 2
do
    for noise in "aggre"
    do
        for dual_momentum in 0.1 0.5 0.9
        do
            for delta in 0.2
            do
                for huber_a in 20.0
                do
                    for epsilon in 0.6 0.5
                    do
                        python3 main_pd.py --u_init 0 --dual_lr 1.0 --dual_momentum $dual_momentum --perturbation_lr 1.0 --dataset cifar10 --noise_type $noise --is_human --seed $seed --wandb_log --huber_a $huber_a --huber_delta $delta --epsilon $epsilon
                    done
                done
            done
        done
    done
done