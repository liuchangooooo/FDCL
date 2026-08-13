from DIVO.evaluator.pusht_evaluator import (
    MujocoPushTEvaluator)

def get_evaluator(**evaluator_cfg):
    target = evaluator_cfg['_target_']
    if target == 'mujoco_pusht':
        evaluator = MujocoPushTEvaluator(**evaluator_cfg)
    else:
        raise NotImplementedError(f"Evaluator type {target} not implemented.")

    return evaluator