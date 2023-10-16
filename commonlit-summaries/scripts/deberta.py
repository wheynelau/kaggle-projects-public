
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ROOT_FOLDER = "./data"
CACHE_DIR = "./.cache"
ADAPTER_FOLDER = "./adapter/deberta-v3-large-v2"
ADAPTER_NAME = 'r32-a64-1000steps-all-prompts'
TRAIN_BATCH = 1
EVAL_BATCH = 4
ACCUM_BATCH = 16
LORA_RANK = 32
LORA_ALPHA = 64
STEPS = 1000
NUM_WORKERS = 0
LR = 1e-3
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
FROZEN= False
EPOCHS = 50
PEFT = False
TRAIN = True
PREDICT = True
LOAD = None
os.environ['WANDB_PROJECT'] = 'kaggle-summaries-scoring'

tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2"
                                          , cache_dir = CACHE_DIR)
model = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/reward-model-deberta-v3-large-v2",
                                                           cache_dir = CACHE_DIR,
                                                            ignore_mismatched_sizes=True,
                                                            problem_type = 'regression',
                                                            num_labels = 2)


if PEFT:
    peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.15,
    # target_modules = ['q','v'],
    )
    p_model = get_peft_model(model,peft_config,ADAPTER_NAME)
    p_model.print_trainable_parameters()
else:
    p_model = model

prompt_train = pd.read_csv(os.path.join(ROOT_FOLDER, "prompts_train.csv"))
summaries_train = pd.read_csv(os.path.join(ROOT_FOLDER,"summaries_train.csv"))
combined = prompt_train.merge(summaries_train, on='prompt_id')

def map_prompts(dataframe):
    return f"""Instruction: \n
    {dataframe['prompt_question']}\n\n
    Summarise: {dataframe['prompt_text']}\n\n
    """

combined['input'] = combined.apply(map_prompts, axis=1)
combined['output'] = combined['text']

class SummaryDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx,-4:].values

def collate_fn(batch):
    input_text = [x[2] for x in batch]
    output_text = [x[3] for x in batch]
    content = [x[0] for x in batch]
    wording = [x[1] for x in batch]
    tokenized_input = tokenizer(input_text,output_text,padding=True, return_tensors = 'pt')
    tokenized_input['labels'] = torch.tensor(list(zip(content, wording))).float()
    return tokenized_input

from sklearn.model_selection import train_test_split
train_df = combined[combined['prompt_id'] != '814d6b']
# train_df, test_df = train_test_split(combined, test_size= 0.2,random_state=42,stratify = combined['prompt_id'])
# train_df.to_csv(os.path.join(ROOT_FOLDER, "train.csv"))
val_df = combined[combined['prompt_id'] == '814d6b']


train_dataset = SummaryDataset(train_df)
val_dataset = SummaryDataset(val_df)
val_dataset = Subset(val_dataset, list(range(0,32)))
from transformers import Trainer, TrainingArguments

training_arguments = TrainingArguments(
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=ACCUM_BATCH // TRAIN_BATCH,
        eval_accumulation_steps=ACCUM_BATCH // EVAL_BATCH,
        max_steps=STEPS,
        learning_rate=LR_BACKBONE,
        lr_scheduler_type='cosine',
        bf16=True,
        warmup_steps=25,
        eval_delay=5,
        evaluation_strategy='steps',
        eval_steps=50,
        logging_strategy = 'steps',
        logging_steps=50,
        save_strategy = 'steps',
        save_steps=50,
        save_total_limit = 1,
        load_best_model_at_end=True,
        run_name= 'full-FT-OOD',
        output_dir="./models",
    )
trainer = Trainer(
    model = p_model,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    data_collator = collate_fn,
    args = training_arguments
)

# 1. Get optimizer class and arguments
optimizer_cls, optimizer_kwargs = trainer.get_optimizer_cls_and_kwargs(training_arguments)

# 2. Group model parameters
score_params = [p for n, p in model.named_parameters() if 'model.classifier' in n]
other_params = [p for n, p in model.named_parameters() if 'model.classifier' not in n]

# Prepare optimizer parameter groups
optimizer_grouped_parameters = [
    {'params': score_params, **optimizer_kwargs},
    {'params': other_params, **optimizer_kwargs}
]

optimizer_grouped_parameters[0]['lr'] = LR
optimizer_grouped_parameters[1]['lr'] = LR_BACKBONE
optimizer = optimizer_cls(optimizer_grouped_parameters)
trainer.optimizer = optimizer
trainer.train()

p_model.save_pretrained('./models')