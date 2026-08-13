from DIVO.env.pusht.mujoco.pusht_mj_rod import PushT_mj_rod

def get_pusht_env(**class_args):
    target = class_args['_target_']
    if 'generate_dataset' in class_args:
        generate_dataset = class_args['generate_dataset']
    else:
        generate_dataset = False
    if 'motion_pred' in class_args:
        motion_pred = class_args['motion_pred']
    else:
        motion_pred = False
    if 'eval' in class_args:
        eval = class_args['eval']
    else:
        eval = False
    if 'obstacle_shape' in class_args:
        obstacle_shape = class_args['obstacle_shape']
    else:
        obstacle_shape = 'box'
    if 'obstacle_num' in class_args:
        obstacle_num = class_args['obstacle_num']
    else:
        obstacle_num = 1
    if 'dynamics_randomization' in class_args:
        dynamics_randomization = class_args['dynamics_randomization']
    else:
        dynamics_randomization = False

    if target == 'pusht_mujoco':
        env = PushT_mj_rod(
            obstacle=class_args["obstacle"],
            obstacle_num=obstacle_num,
            obstacle_size=class_args["obstacle_size"],
            obstacle_shape=obstacle_shape,
            obstacle_dist=class_args["obstacle_dist"],
            record_frame=False,
            action_dim=class_args["action_dim"],
            obs_dim=class_args["obs_dim"],
            action_scale=class_args["action_scale"],
            NUM_SUBSTEPS=class_args["NUM_SUBSTEPS"],
            action_reg=class_args["action_reg"],
            reg_coeff=class_args["reg_coeff"],
            generate_dataset=generate_dataset,
            motion_pred=motion_pred,
            eval=eval,
            dynamics_randomization=dynamics_randomization,
        )
    else:
        raise NotImplementedError(f"Env type {target} not implemented.")

    return env 