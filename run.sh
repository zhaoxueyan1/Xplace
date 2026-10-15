LD_PRELOAD=/nfs/home/shenxurong/miniconda3/envs/opt-gpu/lib/libstdc++.so.6 \
python main.py \
    --dataset ispd2015_fix \
    --run_all True \
    --detail_placement True \
    --use_cell_inflate True \
    --final_route_eval True