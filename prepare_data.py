from datasets import load_dataset, Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

num_samples = 1000
max_article_length = 512
max_summary_length = 100
device = "cuda:1" if torch.cuda.is_available() else "cpu"


dataset = load_dataset("Someman/news_nepali", split='train')


weak_model_name = "universalml/NepaliGPT-2.0"
weak_tokenizer = AutoTokenizer.from_pretrained(weak_model_name)
weak_model = AutoModelForCausalLM.from_pretrained(weak_model_name).to(device)

weak_summarizer = pipeline(
    "text-generation",
    model=weak_model,
    tokenizer=weak_tokenizer,
    device= device,
)


dpo_records = []
for sample in tqdm(dataset):
    article = sample["article"][:max_article_length]
    prompt = f"Please summarize the following article in Nepali:\n{article}"
    chosen = sample["article_summary"]

    try:
        output = weak_summarizer(
            prompt,
            max_new_tokens=max_summary_length,
            min_length=20,
            do_sample=True,
        )
        rejected = output[0]["generated_text"]
        rejected = rejected[len(prompt):].strip()
    except Exception as e:
        print(f"Error generating rejected summary: {e}")
        rejected = prompt[:30] + "..."

    dpo_records.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
        
    })

dpo_dataset = Dataset.from_list(dpo_records)

dpo_dataset.save_to_disk("nepali_news_dpo_dataset")
print("DPO dataset saved to 'nepali_news_dpo_dataset'")
