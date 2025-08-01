
export GPU=1
c
index=0
for value in "${strArray[@]}"
do 
    index=$((index + 1))
    echo "Index $index: $value"
    echo "Index $index: ${task_strArray[index-1]}"
    CUDA_VISIBLE_DEVICES=$GPU  python -u sample_mul_concept.py  \
        --prompt "a <new1> dog  and a <new3> cat sitting on the grass, blue sky"\
        --n_samples 1 \
        --n_iter 10 \
        --ddim_steps 50 \
        --delta_ckpt "/your/path/to/ckpt"  \
        --ckpt sd-v1-5-emaonly.ckpt \
        --concept "mul-dog-cat"\
        --concepts "dog" "cat"\
        --task_id 100\
        --task_prompt "photo of a <new1> dog" "photo of a <new3> cat" \
        --region_prompt  "photo of a <new1> dog" "photo of a <new3> cat" \
        --skip_grid \
        --mask_path logs/dog-cat-mul
done