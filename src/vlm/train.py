from contextlib import nullcontext
import torch
import torch.distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.amp.grad_scaler import GradScaler
import time
from checkpointing import save_model
import os
import logging
from logging import getLogger, Formatter, StreamHandler


log_formatter = Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
console_handler = StreamHandler()
console_handler.setFormatter(log_formatter)

# sets up persistent logs; already logged in wandb though
# fileHandler = logging.FileHandler("nate.log")
# fileHandler.setFormatter(log_formatter)

logger = getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
# logger.addHandler(fileHandler)

# logger.info(os.getenv("WANDB_API_KEY", "nate"))


# so far only works when launching with fsdp
def train(
    model,
    optimizer,
    scheduler,
    train_dl_factory,
    train_config,
    eval_dl=None,
    wandb_run=None,
):
    logger.info("training")

    c = train_config

    # tf32 (only on ampere)
    compute_capability_major, compute_capability_minor = (
        torch.cuda.get_device_capability()
    )
    if c.tf32:
        if compute_capability_major >= 8:
            logger.info("Enabling lower precision tf32 computation")
            torch.set_float32_matmul_precision = "medium"  # | "highest", "high"
        else:
            logger.info(
                "Device compute capability is: {compute_capability_major}.{compute_capability_minor} -- running in fp32"
            )

    # mixed precision scalers
    if c.fp16 and c.fsdp:
        scaler = ShardedGradScaler()
    elif c.fp16 and not c.fsdp:
        scaler = GradScaler()

    # handles moving inputs to right dtype
    autocast = (
        torch.autocast(device_type="cuda") if c.fp16 or c.bfloat16 else nullcontext()
    )

    rank = dist.get_rank()
    _world_size = dist.get_world_size()
    model.train()

    start_time = time.time()
    prev_batch = {}

    for i in range(c.n_epochs):
        train_dl = train_dl_factory()
        total_len = len(train_dl)

        for e, batch in enumerate(train_dl):
            optimizer.zero_grad()

            tik = time.time()
            if batch is None:  # if we can't download imgs use previous data
                print(f"Encountered empty batch on rank {dist.get_rank()}")
                batch = prev_batch
            else:
                prev_batch = batch

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(model.device)

            # autocast forward pass
            with autocast:
                loss = model(**batch, output_attentions=False)["loss"]
                loss = loss / c.gradient_accumulation_steps

            if c.fp16:
                scaler.scale(loss).backward()  # type: ignore
                if (e + 1) % c.gradient_accumulation_steps == 0:
                    if c.grad_clip is not None:
                        scaler.unscale_(optimizer)
                        if c.fsdp:
                            model.clip_grad_norm_(c.grad_clip)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), c.grad_clip
                            )
                        pass

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (e + 1) % c.gradient_accumulation_steps == 0:
                    if c.grad_clip is not None:
                        if c.fsdp:
                            model.clip_grad_norm_(c.grad_clip)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), c.grad_clip
                            )
                    optimizer.step()
                    optimizer.zero_grad()

            # step lr
            scheduler.step()

            # logging stuff
            if (e + 1) % c.log_steps == 0:
                # optimization, only all reduce on log steps
                dt = time.time() - tik

                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()

                if os.environ["LOCAL_RANK"] == "0":
                    logger.info(
                        f"iter: {e+1}/{total_len} loss: {c.gradient_accumulation_steps*loss.item():.2f} mm_projector_lr: {scheduler.get_last_lr()[0]:.6f} llm_lr: {scheduler.get_last_lr()[1]:.6f} [{(time.time()-start_time)/60:.2f} < {(dt*total_len/60):.2f}, {dt:.2f}s/it]"
                    )

            if wandb_run is not None:
                data = {
                    "loss": c.gradient_accumulation_steps * loss.item(),
                    "ppl": 2 ** (c.gradient_accumulation_steps * loss.item()),
                    "mm_connector_lr": scheduler.get_last_lr()[0],
                    "llm_lr": scheduler.get_last_lr()[1],
                }

                if rank == 0:
                    wandb_run.log(data)

            if (e + 1) % c.save_steps == 0:
                dist.barrier()
                save_dir = os.path.join(c.save_path, f"epoch{i+1}_step{e+1}")
                save_model(model, save_dir=save_dir, rank=dist.get_rank())
                dist.barrier()

            # eval
            if c.do_eval:
                assert (
                    c.eval_steps is not None and eval_dl is not None
                ), "Can't run eval without specifying `eval_steps` in train config!"

                if (e + 1) % c.eval_steps == 0:
                    dist.barrier()
                    if dist.get_rank() == 0:
                        print("------------------------")
                        logger.info("running eval...")

                    # run eval
                    eval_loss = run_eval_step(model, eval_dl)

                    # log
                    if wandb_run is not None:
                        wandb_run.log(
                            {"eval_loss": eval_loss},
                            step=(e + 1),
                        )
                    if dist.get_rank() == 0:
                        logger.info(
                            f"iter: {e+1}/{total_len}  eval_loss: {eval_loss:.2f}"
                        )
                        print("------------------------")

        # training done
        dist.barrier()

        # save on epoch end
        if c.save_path is not None:
            save_dir = os.path.join(c.save_path, f"epoch{i+1}_step{total_len}")
            save_model(model, save_dir=save_dir, rank=dist.get_rank())


def run_eval_step(model, eval_dl):
    prev_batch = {}
    losses = []

    # set model to eval
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(eval_dl):
            if batch is None:  # if we can't download imgs use previous data
                logger.info(f"Encountered empty batch on rank {dist.get_rank()}")
                batch = prev_batch
            else:
                prev_batch = batch

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(model.device)

            # forward pass
            out = model(**batch, output_attentions=False)
            loss = out["loss"]

            # all reduce -- might be slow here, could reduce AFTER loop through batch
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size()
            losses.append(loss.detach().cpu().item())

    # set model back to train again
    model.train()

    return sum(losses) / len(losses)
