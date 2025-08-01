export GPU=0
strArray=("dog,duck_toy,cat,backpack,teddybear")
task_strArray=("photo of a <new1> dog" "photo of a <new2> duck_toy" "photo of a <new3> cat" "photo of a <new4> backpack" "photo of a <new5> teddybear")
index=0
for value in "${strArray[@]}"
do 
    index=$((index + 1))
    echo "Index $index: $value"
    echo "Index $index: ${task_strArray[index-1]}"
    CUDA_VISIBLE_DEVICES=$GPU python -u sample_single_concept.py  \
        --from-file "/customconcept101/5task_prompts/$value.txt" \
        --n_samples 4 \
        --n_iter  1\
        --ddim_steps 50 \
        --delta_ckpt "/your/path/to/ckpt"  \
        --ckpt sd-v1-5-emaonly.ckpt  \
        --concept "$value" \
        --task_id  100\
        --task_prompt "${task_strArray[$index-1]}" \
        --skip_grid
done
