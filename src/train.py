import os
import torch
import torch.distributed as dist
import time
from checkpointing import save_model


def run_eval_step(model, eval_dl):
    prev_batch = {}
    losses = []

    # set model to eval
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(eval_dl):
            if batch is None:  # if we can't download imgs use previous data
                print(f"Encountered empty batch on rank {dist.get_rank()}")
                batch = prev_batch
            else:
                prev_batch = batch

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(model.device)

            # forward pass
            out = model(**batch, output_attentions=False)
            loss = out["loss"]

            # all reduce
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size()
            losses.append(loss.detach().cpu().item())

    # set model back to train again
    model.train()

    return sum(losses) / len(losses)


def train(
    model,
    optimizer,
    scheduler,
    train_dl,
    train_config,
    eval_dl=None,
    wandb_run=None,
):
    print("training")

    c = train_config
    model.train()

    # recover rank
    rank = dist.get_rank()

    total_len = len(train_dl)
    start_time = time.time()
    prev_batch = {}

    # FIXME: we've consumed the dataloader already here in first epoch; copy it or something
    for i in range(c.n_epochs):
        for e, batch in enumerate(train_dl):
            tik = time.time()
            if batch is None:  # if we can't download imgs use previous data
                print(f"Encountered empty batch on rank {dist.get_rank()}")
                batch = prev_batch
            else:
                # potentially is just a single sample
                prev_batch = batch

            # print(batch.keys())
            # print(batch["input_ids"].shape)

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(model.device)  # assumes gpu training

            # forward pass
            out = model(**batch, output_attentions=False)
            loss = out["loss"]

            loss = loss / c.gradient_accumulation_steps
            loss.backward()

            if (e + 1) % c.gradient_accumulation_steps == 0:
                if c.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            if (e + 1) % c.log_steps == 0:
                dt = time.time() - tik

                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()

                if os.environ["LOCAL_RANK"] == "0":
                    print(
                        f"iter: {e+1}/{total_len}  loss: {c.gradient_accumulation_steps*loss.item():.2f}  lr: {scheduler.get_last_lr()[0]:.6f} [{(time.time()-start_time)/60:.2f} < {(dt*total_len/60):.2f}, {dt:.2f}s/it]"
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
                save_dir = os.path.join(c.save_path, f"step{e+1}")
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
                        print("running eval...")

                    # run eval
                    eval_loss = run_eval_step(model, eval_dl)

                    # log
                    if wandb_run is not None:
                        wandb_run.log(
                            {"eval_loss": eval_loss},
                            step=(e + 1),
                        )
                    if dist.get_rank() == 0:
                        print(f"iter: {e+1}/{total_len}  eval_loss: {eval_loss:.2f}")
                        print("------------------------")

    # training done
    dist.barrier()

    # save on epoch end
    if c.save_path is not None:
        save_dir = os.path.join(c.save_path, f"step{total_len}")
        save_model(model, save_dir=save_dir, rank=dist.get_rank())

        def foo(i: int) -> float:
            return 1.0

        foo("nate")
