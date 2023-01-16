CUDA_VISIBLE_DEVICES=1
for seed in 0
do
    for noise in "aggre"
    do
            for delta in 0.2 0.5 1.0
            do
                for huber_a in 10.0 2.0 20.0
                do
                for epsilon in 0.8 0.6
                    do
                    python3 main_pd.py --dataset cifar10 --noise_type $noise --is_human --seed $seed --wandb_log --huber_a $huber_a --huber_delta $delta --epsilon $epsilon
                    done
            done
        done
    done
done