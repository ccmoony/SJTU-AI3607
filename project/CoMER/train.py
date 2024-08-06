from pytorch_lightning.plugins.training_type import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI

from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER

cli = LightningCLI(
    LitCoMER,
    CROHMEDatamodule,
    save_config_overwrite=True,
    trainer_defaults={
            "plugins": [{
        "class_path": "pytorch_lightning.plugins.training_type.DDPPlugin",
        "init_args": {
            "find_unused_parameters": False
        }
    }],
    "resume_from_checkpoint": "/home/bml/storage/mnt/v-c615a05aea3047cd/org/users/zouzichen/CoMER/CoMER/lightning_logs/version_50/checkpoints/epoch=84-step=31959-val_ExpRate=0.4763.ckpt"
    },
 
)
