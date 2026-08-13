import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from DIVO.workspace.base_workspace import BaseWorkspace

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.learn()

if __name__ == "__main__":
    main()