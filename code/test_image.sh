model=sbm
nc=128 # number of clusters
st=BMS+SB
sw=1e-1 # sparsity regularizer
k0=8
k5=16
k1=32 #kk
k2=64
k3=128

for task in image
do
    nohup python3 run_tasks.py --task $task --model $model --num_clusters $nc --sbm_type $st --sparsity_weight $sw --dota_k $k0 &
done 


