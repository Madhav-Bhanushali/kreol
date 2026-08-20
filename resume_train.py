"""Resume training from the trained 10-epoch mfe LoRA adapter for 5 more epochs.

This is a comparison experiment: "does more training reduce loss further?"
Settings are IDENTICAL to the first run (train.py) except:
  - starts from the trained adapter (new_lang_adapter) instead of fresh LoRA init
  - num_epochs = 5 (was 10)
  - outputs to mfe_output/extended_15ep/ (original 10-ep output untouched)
  - reuses the existing preprocessed .pt data (preprocess=False)

NOTE: a fresh optimizer/scheduler is created (no resume_from_checkpoint), so the
linear LR schedule restarts at 1e-4 and anneals over the 5 epochs.
"""
import os
import sys
import torch
from transformers import Trainer, TrainingArguments

from src.config import TrainConfig
from src.dataset import ChatterboxDataset, data_collator_standart
from src.model import resize_and_load_t3_weights, ChatterboxTrainerWrapper
from src.utils import setup_logger, check_pretrained_models

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.t3.t3 import T3
from peft import PeftModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = setup_logger("ChatterboxResume")


def main():
    cfg = TrainConfig()
    # --- experiment overrides (everything else stays identical) ---
    cfg.output_dir = os.path.join(cfg.output_dir, "extended_15ep")
    cfg.num_epochs = 5
    cfg.preprocess = False
    adapter_dir = os.path.join(os.path.dirname(cfg.output_dir), "new_lang_adapter")

    if not check_pretrained_models(mode="chatterbox"):
        sys.exit(1)
    if not os.path.exists(adapter_dir):
        logger.error(f"Adapter not found: {adapter_dir}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info("Loading original model to extract weights...")
    tts_engine_original = ChatterboxMultilingualTTS.from_local(
        cfg.model_dir, device="cpu", t3_model=cfg.t3_model
    )
    pretrained_t3_state_dict = tts_engine_original.t3.state_dict()
    original_t3_config = tts_engine_original.t3.hp

    logger.info(f"Creating new T3 model with vocab size: {cfg.new_vocab_size}")
    new_t3_config = original_t3_config
    new_t3_config.text_tokens_dict_size = cfg.new_vocab_size
    if hasattr(new_t3_config, "use_cache"):
        new_t3_config.use_cache = False
    else:
        setattr(new_t3_config, "use_cache", False)

    new_t3_model = T3(hp=new_t3_config)
    new_t3_model = resize_and_load_t3_weights(
        new_t3_model, pretrained_t3_state_dict, new_token_init_row=cfg.fr_token_id
    )
    del tts_engine_original, pretrained_t3_state_dict

    tts_engine_new = ChatterboxMultilingualTTS.from_local(
        cfg.model_dir, device="cpu", t3_model=cfg.t3_model
    )
    tts_engine_new.t3 = new_t3_model

    logger.info("Freezing S3Gen and VoiceEncoder...")
    for param in tts_engine_new.ve.parameters():
        param.requires_grad = False
    for param in tts_engine_new.s3gen.parameters():
        param.requires_grad = False

    logger.info("Freezing all T3 params, then loading trained adapter...")
    for param in tts_engine_new.t3.parameters():
        param.requires_grad = False

    tts_engine_new.t3 = PeftModel.from_pretrained(
        tts_engine_new.t3, adapter_dir, is_trainable=True
    )
    tts_engine_new.t3.print_trainable_parameters()

    logger.info("Initializing Dataset...")
    train_ds = ChatterboxDataset(cfg)

    model_wrapper = ChatterboxTrainerWrapper(tts_engine_new.t3)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_strategy="steps",
        logging_steps=25,
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to=["tensorboard"],
        fp16=False,
        bf16=True,
        save_total_limit=cfg.save_total_limit,
        gradient_checkpointing=False,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator_standart,
    )

    logger.info("Starting Extended Training Loop...")
    trainer.train()

    logger.info("Training complete. Saving extended adapter...")
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, "new_lang_adapter")
    tts_engine_new.t3.save_pretrained(save_path)
    logger.info(f"Extended adapter saved to: {save_path}")


if __name__ == "__main__":
    main()