def get_env_class(**class_args):
    target = class_args['_target_']
    if 'pusht' in target:
        from DIVO.env.pusht import get_pusht_env
        env = get_pusht_env(**class_args)
    elif 'ant_nav' in target:
        from DIVO.ant.factory import get_ant_env
        env = get_ant_env(**class_args)

    else:
        raise NotImplementedError(f"Env type {target} not implemented.")

    return env
    
