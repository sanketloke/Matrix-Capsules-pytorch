#PBS -l nodes=1:ppn=2
#PBS -l walltime=50:00:00
#PBS -q p100_normal_q ##or p100_dev_q for short jobs
#PBS -W group_list=newriver
#PBS -A vllab_2017
source activate py36
module load cuda/8.0.44
module load cudnn/6.0
cd /home/sloke/oml/Matrix-Capsules-pytorch
#python3 train.py "smallNORB"
echo $CUDA_VISIBLE_DEVICES
#python train.py "smallNORB"
#python eval_accuracy.py "smallNORB"

python train.py -batch_size=64 -lr=2e-2 -num_epochs=50 -r=1 -print_freq=1 -dataset='MNIST' -dist_freq=2 -fname=mnist_test -model=CNN

