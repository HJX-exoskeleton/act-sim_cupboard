# ACT: Action Chunking with Transformers

### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


# New Task (Mujoco simulation)
# Simultaneous Grad Task
    python3 record_sim_episodes.py \
    --task_name sim_simultaneous_grad_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

# Invert Transfer Cube Task -- Upgrade
    python3 record_sim_episodes.py \
    --task_name sim_invert_transfer_cube_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

# Invert Open The Curtain Task 
    python3 record_sim_episodes.py \
    --task_name sim_invert_open_the_curtain_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

# Cupboard Task 
    python3 record_sim_episodes.py \
    --task_name sim_cupboard_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

# Invert Building Blocks Task 
    python3 record_sim_episodes.py \
    --task_name sim_invert_building_blocks_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the episode after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

sim_simultaneous_grad_scripted: visualize
![image](https://github.com/M3-song/aloha-mujoco-sim/assets/156507453/92da7d14-18b6-48b2-ad64-ce7d598e06d9)

sim_invert_transfer_cube_scripted: visualize
![image](https://github.com/M3-song/aloha-mujoco-sim/assets/156507453/b05238d7-fd04-4779-830d-26a9a8394024)

sim_invert_open_the_curtain_scripted: visualize
![image](https://github.com/M3-song/aloha-mujoco-sim/assets/156507453/79cefc73-2abc-4698-8963-ad54f6bbb14b)

sim_cupboard_scripted: visualize
![image](https://github.com/M3-song/aloha-mujoco-sim/assets/156507453/facf2764-aa51-4ac5-a813-2a755cafb46f)

sim_invert_building_blocks_scripted: visualize
![image](https://github.com/user-attachments/assets/40b3b0d9-6b9b-4650-9e5b-e6b6219f7d72)


To train ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py \
    --task_name sim_simultaneous_grad_scripted \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 \
    --seed 0

To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

For real-world data where things can be harder to model, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.
Please refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing) for more info.


