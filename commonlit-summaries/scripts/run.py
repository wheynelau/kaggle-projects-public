from transformers import LlamaForSequenceClassification, LlamaTokenizer, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded",
                                          cache_dir = "./llama2",
                                          load_in_8bit= True,
                                          use_auth_token=True)
model = LlamaForSequenceClassification.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded",
                                                       cache_dir = "../llama2",
                                                       load_in_8bit = True,
                                                       problem_type = 'regression',
                                                       use_auth_token=True, num_labels = 2)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
lora_alpha = 16
lora_dropout = 0.1
lora_r = 32

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="SEQ_CLS"
)

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
ROOT_FOLDER = '../data'
prompt_train = pd.read_csv(os.path.join(ROOT_FOLDER, "prompts_train.csv"))
summaries_train = pd.read_csv(os.path.join(ROOT_FOLDER,"summaries_train.csv"))
combined = prompt_train.merge(summaries_train, on='prompt_id')

def map_prompts(dataframe):
    return f"""Instruction: \n
    {dataframe['prompt_question']}\n\n
    Summarise: {dataframe['prompt_text']}\n\n
    Rate the given summary: {dataframe['text']}
    """

combined['input'] = combined.apply(map_prompts, axis=1)
# combined['output'] = combined['text']

class SummaryDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx,-3:].values

def collate_fn(batch):
    input_text = [x[2] for x in batch]
    content = [x[0] for x in batch]
    wording = [x[1] for x in batch]
    tokenized_input = tokenizer(input_text,padding=True, return_tensors = 'pt')
    tokenized_input['labels'] = torch.tensor(list(zip(content, wording))).float()
    return tokenized_input

# %%
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# %%
tokenizer.pad_token = tokenizer.eos_token

# %%
for name, module in model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# %%
from sklearn.model_selection import train_test_split
train_df = combined[combined['prompt_id'] != '814d6b']
train_df, test_df = train_test_split(train_df, test_size= 0.2,random_state=42,stratify = train_df['prompt_id'])
train_df.to_csv(os.path.join(ROOT_FOLDER, "train.csv"))
val_df = combined[combined['prompt_id'] == '814d6b']


train_dataset = SummaryDataset(train_df)
val_dataset = SummaryDataset(val_df)

# %%
from transformers import Trainer, TrainingArguments

training_arguments = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        max_steps=400,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=1,
        save_strategy = 'steps',
        save_steps = 100,
        save_total_limit = 1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    )
trainer = Trainer(
    model = model,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    data_collator = collate_fn,
    args = training_arguments
)

# 1. Get optimizer class and arguments
optimizer_cls, optimizer_kwargs = trainer.get_optimizer_cls_and_kwargs(training_arguments)

# 2. Group model parameters
score_params = [p for n, p in model.named_parameters() if 'model.score' in n]
other_params = [p for n, p in model.named_parameters() if 'model.score' not in n]

# Prepare optimizer parameter groups
optimizer_grouped_parameters = [
    {'params': score_params, **optimizer_kwargs},
    {'params': other_params, **optimizer_kwargs}
]

optimizer_grouped_parameters[0]['lr'] = optimizer_kwargs['lr'] * 100
optimizer = optimizer_cls(optimizer_grouped_parameters)
trainer = Trainer(
    model = model,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    data_collator = collate_fn,
    args = training_arguments,
    optimizers=(optimizer, None)
)
trainer.train()

model.save_pretrained('./adapter')
