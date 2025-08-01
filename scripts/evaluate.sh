export GPU=1
# Replace "/your/path/to/ckpt/" with the actual path on your system where the trained model is stored
# Replace "/your/path/to/dataset/" with the path to the dataset you want to evaluate on
# Replace "/your/path/to/prompts/" with the path to the directory containing the prompt templates
# concepts is a list of concepts to evaluate on

CUDA_VISIBLE_DEVICES=$GPU python customconcept101/evaluate.py \
    --sample_root /your/path/to/sample_images/  \
    --target_path /your/path/to/dataset/ \
    --prompt_root /your/path/to/prompts/ \
    --concepts "dog,duck_toy,cat,backpack,teddybear"\
    --name "evaluate"