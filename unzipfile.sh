# download data from huggingface 
# for one file 
# huggingface-cli download ShareGPTVideo/train_video_and_instruction  --include "train_300k/chunk_1.tar.gz" --repo-type dataset --local-dir /mnt/disks/new-disk/LVLM-reasoning/data/videos/train_300k

# for one folder:
#huggingface-cli download ShareGPTVideo/train_video_and_instruction  --include "train_300k/*" --repo-type dataset --local-dir /mnt/disks/new-disk/LVLM-reasoning/data/videos/train_300k


for z in data/data/*.zip
do 
    unzip -q "$z" -d "data/InternVL-Chat-V1-2-SFT-Data"; 
    rm "$z"
done

# video_zip_dir=videos/train_300k
# for chunk_path in "$video_zip_dir"/chunk_*; do
#     tar -xzf ${chunk_path} -C ./sharegptvideo
#     rm "$chunk_path"
# done
