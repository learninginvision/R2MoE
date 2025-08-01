
export GPU=0
strArray=("dog,duck_toy,cat,backpack,teddybear")
task_strArray=("photo of a <new1> dog" "photo of a <new2> duck_toy" "photo of a <new3> cat" "photo of a <new4> backpack" "photo of a <new5> teddybear")
promt_strArray=("<new1> dog" "<new2> duck_toy" "<new3> cat" "<new4> backpack" "<new5> teddybear")
index=0

for value in "${strArray[@]}"
do 
    index=$((index + 1))
    echo "Index $index: $value"
    echo "Index $index: ${task_strArray[index-1]}"
    CUDA_VISIBLE_DEVICES=$GPU python -u sample_single_concept.py  \
        --prompt  "${promt_strArray[index-1]}"\
        --n_samples 1 \
        --n_iter  25\
        --ddim_steps 50 \
        --ckpt sd-v1-5-emaonly.ckpt \
        --concept "$value" \
        --task_id 200 \
        --task_prompt "${task_strArray[$index-1]}" \
        --skip_grid
done