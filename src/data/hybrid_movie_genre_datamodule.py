from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class HybridMovieGenreDataModule(LightningDataModule):
    def __init__(
        self,
        train_set,
        val_set,
        test_set,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False
    ) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.hparams.train_set
            self.data_val = self.hparams.val_set
            self.data_test = self.hparams.test_set

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

############################################################### TEST ###############################################################
import hydra
from omegaconf import DictConfig
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # test instance
    dm: LightningDataModule = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    dm.setup()
    
    # test dataset
    print(f"Length of train dataset: {len(dm.data_train)}")
    print(f"Length of val dataset: {len(dm.data_val)}")
    print(f"Length of test dataset: {len(dm.data_test)}\n")
    
    # test dataloader
    print(f"Length of train dataloader: {len(dm.train_dataloader())}")
    print(f"Length of val dataloader: {len(dm.val_dataloader())}")
    print(f"Length of test dataloader: {len(dm.test_dataloader())}\n")
    
    # test batch
    (ratings_general, ratings_age, ratings_occupation), title_input_ids, title_attn_mask, img_tensor, genre_tensor = next(iter(dm.val_dataloader()))
    
    print(f"Length of one batch (batch size): {len(genre_tensor)}")
    print(f"Shape of ratings_general: {ratings_general.size()}")
    print(f"Shape of ratings_age: {ratings_age.size()}")
    print(f"Shape of ratings_occupation: {ratings_occupation.size()}")
    print(f"Shape of title_input_ids: {title_input_ids.size()}")
    print(f"Shape of title_attn_mask: {title_attn_mask.size()}")
    print(f"Shape of img_tensor: {img_tensor.size()}")
    print(f"Shape of output: {genre_tensor.size()}")
    
    print(ratings_general, ratings_age, ratings_occupation)

if __name__ == "__main__":
    main()
############################################################### TEST ###############################################################
