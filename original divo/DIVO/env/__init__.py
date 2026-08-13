from DIVO.env.pusht import get_pusht_env

def get_env_class(**class_args):
    target = class_args['_target_']
    if 'pusht' in target:
        env = get_pusht_env(**class_args)

    else:
        raise NotImplementedError(f"Env type {target} not implemented.")

    return env
    