
# when visualizing the output motion saved in SOMA format
cd /home/jungbin_cho/kimodo && PYOPENGL_PLATFORM=egl /home/jungbin_cho/miniforge3/envs/kimodo/bin/python -m kimodo.scripts.visualize \
    /weka/jungbin/seed/soma_uniform_motions_20fps/210531/jump_and_land_heavy_001__A001.npz \
    --view general,top,front --fps 20 --output ./vis_out


