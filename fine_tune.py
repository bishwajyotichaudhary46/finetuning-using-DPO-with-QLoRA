import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from torch.amp import autocast, GradScaler
import signal
import sys
import bitsandbytes as bnb


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_NCCL_TIMEOUT"] = "1800"
os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1048576"


model_name = "universalml/NepaliGPT-2.0"
output_dir = "./dpo_nepali_results"
num_epochs = 1
batch_size_per_gpu = 1
accumulation_steps = 2
learning_rate = 1e-5
beta = 0.1
max_length = 256
max_prompt_length = 128
eval_steps = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def signal_handler(sig, frame):
    print("Received SIGINT, saving checkpoint and exiting...")
    if rank == 0:
        checkpoint_dir = os.path.join(output_dir, "interrupt_checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.module.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved interrupt checkpoint to {checkpoint_dir}")
    cleanup_ddp()
    sys.exit(0)


def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def log_vram(rank, stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(rank) / 1024**3
        reserved = torch.cuda.memory_reserved(rank) / 1024**3
        print(f"Rank {rank} {stage} - VRAM: {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved")

def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params} || Total params: {total_params} || % Trainable: {trainable_params / total_params * 100:.2f}%")


def main():
    global rank, local_rank, model, tokenizer
    try:
       
        rank, local_rank = setup_ddp()
        signal.signal(signal.SIGINT, signal_handler)
       
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_cache=False,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        ).to(local_rank)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        log_vram(local_rank, "After model load")

       
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            use_rslora=True
        )
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
        model = DDP(model, device_ids=[local_rank])
        log_vram(local_rank, "After DDP")

        
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_cache=False,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        ).to(local_rank)
        ref_model.eval()
        log_vram(local_rank, "After ref_model load")

      
        if rank == 0:
            dataset = load_from_disk("nepali_news_dpo_dataset")
            print(f"Total samples: {len(dataset)}")
            train_val_split = dataset.train_test_split(test_size=0.2, seed=42)
            val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=42)
            train_dataset = train_val_split["train"]
            val_dataset = val_test_split["train"]
            test_dataset = val_test_split["test"]
            print(f"Train samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")
            print(f"Test samples: {len(test_dataset)}")
            
            os.makedirs(os.path.join(output_dir, "temp"), exist_ok=True)
            train_dataset.save_to_disk(os.path.join(output_dir, "temp/train"))
            val_dataset.save_to_disk(os.path.join(output_dir, "temp/val"))
            test_dataset.save_to_disk(os.path.join(output_dir, "temp/test"))
        
        dist.barrier()
        train_dataset = load_from_disk(os.path.join(output_dir, "temp/train"))
        val_dataset = load_from_disk(os.path.join(output_dir, "temp/val"))
        test_dataset = load_from_disk(os.path.join(output_dir, "temp/test"))

       
        def preprocess_dpo(example):
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]
            prompt_inputs = tokenizer(
                prompt,
                max_length=max_prompt_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            chosen_inputs = tokenizer(
                chosen,
                max_length=max_length - max_prompt_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            rejected_inputs = tokenizer(
                rejected,
                max_length=max_length - max_prompt_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return {
                "prompt_ids": prompt_inputs["input_ids"].squeeze(),
                "prompt_mask": prompt_inputs["attention_mask"].squeeze(),
                "chosen_ids": chosen_inputs["input_ids"].squeeze(),
                "chosen_mask": chosen_inputs["attention_mask"].squeeze(),
                "rejected_ids": rejected_inputs["input_ids"].squeeze(),
                "rejected_mask": rejected_inputs["attention_mask"].squeeze()
            }

        train_dataset = train_dataset.map(preprocess_dpo, num_proc=1)
        val_dataset = val_dataset.map(preprocess_dpo, num_proc=1)
        test_dataset = test_dataset.map(preprocess_dpo, num_proc=1)

        train_dataset.set_format("torch")
        val_dataset.set_format("torch")
        test_dataset.set_format("torch")

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size_per_gpu,
            sampler=train_sampler,
            shuffle=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size_per_gpu,
            sampler=val_sampler,
            shuffle=False
        )

        
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scaler = GradScaler('cuda')

       
        def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
            policy_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            losses = -F.logsigmoid(beta * (policy_logratios - ref_logratios))
            return losses.mean()

        def compute_preference_accuracy(policy_chosen_logps, policy_rejected_logps):
            correct = (policy_chosen_logps > policy_rejected_logps).float()
            return correct.mean().item()

        
        metrics_file = os.path.join(output_dir, "training_metrics.csv")
        metrics = []

        
        model.train()
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            total_train_loss = 0
            train_steps = 0
            accum_loss = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Rank {rank} Training", disable=rank != 0)):
                try:
                    prompt_ids = batch["prompt_ids"].to(local_rank)
                    prompt_mask = batch["prompt_mask"].to(local_rank)
                    chosen_ids = batch["chosen_ids"].to(local_rank)
                    chosen_mask = batch["chosen_mask"].to(local_rank)
                    rejected_ids = batch["rejected_ids"].to(local_rank)
                    rejected_mask = batch["rejected_mask"].to(local_rank)

                    chosen_inputs = torch.cat([prompt_ids, chosen_ids], dim=1)
                    chosen_attention_mask = torch.cat([prompt_mask, chosen_mask], dim=1)
                    rejected_inputs = torch.cat([prompt_ids, rejected_ids], dim=1)
                    rejected_attention_mask = torch.cat([prompt_mask, rejected_mask], dim=1)

                    with autocast('cuda'):
                        chosen_outputs = model(chosen_inputs, attention_mask=chosen_attention_mask)
                        rejected_outputs = model(rejected_inputs, attention_mask=rejected_attention_mask)
                        
                        chosen_logits = chosen_outputs.logits[:, :-1].reshape(-1, chosen_outputs.logits.size(-1))
                        chosen_labels = chosen_inputs[:, 1:].reshape(-1)
                        policy_chosen_logps = -F.cross_entropy(chosen_logits, chosen_labels, reduction="mean")
                        
                        rejected_logits = rejected_outputs.logits[:, :-1].reshape(-1, rejected_outputs.logits.size(-1))
                        rejected_labels = rejected_inputs[:, 1:].reshape(-1)
                        policy_rejected_logps = -F.cross_entropy(rejected_logits, rejected_labels, reduction="mean")

                        ref_chosen_outputs = ref_model(chosen_inputs, attention_mask=chosen_attention_mask)
                        ref_rejected_outputs = ref_model(rejected_inputs, attention_mask=rejected_attention_mask)
                        
                        ref_chosen_logits = ref_chosen_outputs.logits[:, :-1].reshape(-1, ref_chosen_outputs.logits.size(-1))
                        ref_chosen_logps = -F.cross_entropy(ref_chosen_logits, chosen_labels, reduction="mean")
                        
                        ref_rejected_logits = ref_rejected_outputs.logits[:, :-1].reshape(-1, ref_rejected_outputs.logits.size(-1))
                        ref_rejected_logps = -F.cross_entropy(ref_rejected_logits, rejected_labels, reduction="mean")

                        loss = dpo_loss(
                            policy_chosen_logps,
                            policy_rejected_logps,
                            ref_chosen_logps,
                            ref_rejected_logps,
                            beta
                        )

                    scaler.scale(loss / accumulation_steps).backward()
                    accum_loss += loss.item() / accumulation_steps

                   
                    if (batch_idx + 1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        total_train_loss += accum_loss
                        train_steps += 1
                        accum_loss = 0
                        
                        if rank == 0 and train_steps % eval_steps == 0:
                            print(f"Step {train_steps}, Train Loss: {total_train_loss / train_steps:.4f}")
                            log_vram(local_rank, f"Step {train_steps}")

                except RuntimeError as e:
                    print(f"Rank {rank} Error in training step {train_steps}: {e}")
                    torch.cuda.empty_cache()
                    continue

            
            total_train_loss_tensor = torch.tensor(total_train_loss, device=local_rank)
            train_steps_tensor = torch.tensor(train_steps, device=local_rank)
            dist.all_reduce(total_train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_steps_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = (total_train_loss_tensor / train_steps_tensor).item()

           
            model.eval()
            total_val_loss = 0
            total_val_acc = 0
            val_steps = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Rank {rank} Validation", disable=rank != 0):
                    try:
                        prompt_ids = batch["prompt_ids"].to(local_rank)
                        prompt_mask = batch["prompt_mask"].to(local_rank)
                        chosen_ids = batch["chosen_ids"].to(local_rank)
                        chosen_mask = batch["chosen_mask"].to(local_rank)
                        rejected_ids = batch["rejected_ids"].to(local_rank)
                        rejected_mask = batch["rejected_mask"].to(local_rank)

                        chosen_inputs = torch.cat([prompt_ids, chosen_ids], dim=1)
                        chosen_attention_mask = torch.cat([prompt_mask, chosen_mask], dim=1)
                        rejected_inputs = torch.cat([prompt_ids, rejected_ids], dim=1)
                        rejected_attention_mask = torch.cat([prompt_mask, rejected_mask], dim=1)

                        with autocast('cuda'):
                            chosen_outputs = model(chosen_inputs, attention_mask=chosen_attention_mask)
                            rejected_outputs = model(rejected_inputs, attention_mask=rejected_attention_mask)
                            
                            chosen_logits = chosen_outputs.logits[:, :-1].reshape(-1, chosen_outputs.logits.size(-1))
                            chosen_labels = chosen_inputs[:, 1:].reshape(-1)
                            policy_chosen_logps = -F.cross_entropy(chosen_logits, chosen_labels, reduction="mean")
                            
                            rejected_logits = rejected_outputs.logits[:, :-1].reshape(-1, rejected_outputs.logits.size(-1))
                            rejected_labels = rejected_inputs[:, 1:].reshape(-1)
                            policy_rejected_logps = -F.cross_entropy(rejected_logits, rejected_labels, reduction="mean")

                            ref_chosen_outputs = ref_model(chosen_inputs, attention_mask=chosen_attention_mask)
                            ref_rejected_outputs = ref_model(rejected_inputs, attention_mask=rejected_attention_mask)
                            
                            ref_chosen_logits = ref_chosen_outputs.logits[:, :-1].reshape(-1, ref_chosen_outputs.logits.size(-1))
                            ref_chosen_logps = -F.cross_entropy(ref_chosen_logits, chosen_labels, reduction="mean")
                            
                            ref_rejected_logits = ref_rejected_outputs.logits[:, :-1].reshape(-1, ref_rejected_outputs.logits.size(-1))
                            ref_rejected_logps = -F.cross_entropy(ref_rejected_logits, rejected_labels, reduction="mean")

                            val_loss = dpo_loss(
                                policy_chosen_logps,
                                policy_rejected_logps,
                                ref_chosen_logps,
                                ref_rejected_logps,
                                beta
                            )

                        total_val_loss += val_loss.item()
                        val_acc = compute_preference_accuracy(policy_chosen_logps, policy_rejected_logps)
                        total_val_acc += val_acc
                        val_steps += 1

                    except RuntimeError as e:
                        print(f"Rank {rank} Error in validation step {val_steps}: {e}")
                        torch.cuda.empty_cache()
                        continue

            
            total_val_loss_tensor = torch.tensor(total_val_loss, device=local_rank)
            total_val_acc_tensor = torch.tensor(total_val_acc, device=local_rank)
            val_steps_tensor = torch.tensor(val_steps, device=local_rank)
            dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_val_acc_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_steps_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = (total_val_loss_tensor / val_steps_tensor).item()
            avg_val_acc = (total_val_acc_tensor / val_steps_tensor).item()

         
            if rank == 0:
                print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Preference Accuracy: {avg_val_acc:.4f}")
                metrics.append({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_preference_accuracy": avg_val_acc
                })
                metrics_df = pd.DataFrame(metrics)
                metrics_df.to_csv(metrics_file, index=False)
                print(f"Saved metrics to {metrics_file}")

               
                checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.module.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint to {checkpoint_dir}")

            model.train()

    
        if rank == 0:
            model.module.save_pretrained("./dpo_nepali_fine_tuned")
            tokenizer.save_pretrained("./dpo_nepali_fine_tuned")
            test_dataset.save_to_disk(os.path.join(output_dir, "test_dataset"))
            print("Training completed.")

    except Exception as e:
        print(f"Rank {rank} Fatal error: {e}")
        raise
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "training_metrics.csv")
    metrics = []
    main()