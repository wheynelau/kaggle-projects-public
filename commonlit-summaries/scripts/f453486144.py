 
# Load model directly
import os
from gc import collect
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from peft import LoraConfig, get_peft_model, TaskType, PromptTuningConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ['WANDB_PROJECT'] = 'kaggle-summaries-scoring'
CACHE_DIR = './.cache'
MODEL_ID = "google/t5-v1_1-large"
ROOT_FOLDER = "./data"
TRAIN_BATCH = 1
EVAL_BATCH = 2
ACCUM_BATCH = 32
LORA_RANK = 16
LORA_ALPHA = 64
CHECKPOINT_NAME = "base-lora-16-{val_loss:.4f}"
# init model

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir = CACHE_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, 
                                              cache_dir = CACHE_DIR,
                                              ignore_mismatched_sizes=True,
                                                problem_type = 'regression',
                                                num_labels = 2)
# Init peft
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    inference_mode=False, 
    r=LORA_RANK, 
    lora_alpha=LORA_ALPHA, 
    lora_dropout=0.1
)

p_model = get_peft_model(model, peft_config)
p_model.print_trainable_parameters()
del model
collect()


prompt_train = pd.read_csv(os.path.join(ROOT_FOLDER, "prompts_train.csv"))
summaries_train = pd.read_csv(os.path.join(ROOT_FOLDER,"summaries_train.csv"))
combined = prompt_train.merge(summaries_train, on='prompt_id')

def map_prompts(dataframe):
    return f"""{dataframe['prompt_question']}\n
    Topic: {dataframe['prompt_title']}\n
    Content text: {dataframe['prompt_text']}\n\n
    """
combined['input'] = combined.apply(map_prompts, axis=1)
combined['output'] = combined['text'].apply(lambda x: "Summary : " + x + " \n\n")

class LitModel(pl.LightningModule):

    def __init__(self, base_model, weight_decay = 1e-4, lr= 1e-4,lr_backbone=1e-6):
        super().__init__()
        self.base_model = base_model
        self.lr_backbone = lr_backbone
        self.lr = lr
        self.weight_decay = weight_decay
        self.base_model.lm_head =  nn.Linear(self.base_model.config.d_model,2)
        self.linear= nn.Linear(32128, 178)
        self.linear2 = nn.linear(178, 13)
        self.linear3 = nn.linear(13, 2)

    def forward(self,**kwargs):
        output = self.base_model(**kwargs)
        pooled = torch.max(output.logits, dim=1).values
        output = self.linear(pooled)
        output = self.linear2(output)
        output = self.linear3(output)
        return F.sigmoid(output)

    def training_step(self, batch, batch_idx=None):
        y = batch.pop('labels')
        y_hat = self.forward(**batch)
        loss = F.l1_loss(y_hat, y)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx=None):
        y = batch.pop('labels')
        y_hat = self.forward(**batch)
        loss = F.l1_loss(y_hat, y)
        self.log('val_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "base_model" in n and ("attention_pooling" not in n and "linear" not in n) and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if ("attention_pooling" in n or "linear" in n) and p.requires_grad
                ],
                "lr": self.lr,  # assuming self.lr > self.lr_backbone
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, weight_decay=self.weight_decay
        )
        return optimizer
    
class SummaryDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx,-4:].values
    
def collate_fn(batch):
    input_text = [x[2] for x in batch]
    content = [x[0] for x in batch]
    wording = [x[1] for x in batch]
    output_text = [x[3] for x in batch]
    tokenized_input = tokenizer(input_text, padding=True, return_tensors = 'pt')
    tokenized_output = tokenizer(output_text, padding=True, return_tensors = 'pt')
    tokenized_input['decoder_input_ids'] = tokenized_output['input_ids']
    tokenized_input['decoder_attention_mask'] = tokenized_output['attention_mask']
    tokenized_input['labels'] = torch.tensor(list(zip(content, wording))).float()
    return tokenized_input
lit = LitModel(p_model, frozen = False)

 
train_df = combined.sample(frac= 0.8)
test_df = combined.drop(train_df.index)

 
from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler()
train_df[['content','wording']] = scl.fit_transform(train_df[['content','wording']])
test_df[['content','wording']] = scl.transform(test_df[['content','wording']])

 
train_dataset = SummaryDataset(train_df)
test_dataset =SummaryDataset(test_df)
train_dataloader= DataLoader(train_dataset, batch_size= TRAIN_BATCH, shuffle=True, collate_fn=collate_fn, num_workers=0)
test_dataloader= DataLoader(test_dataset, batch_size= EVAL_BATCH,shuffle=True,collate_fn=collate_fn, num_workers=0)

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateFinder, ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=CHECKPOINT_NAME,
        verbose=True,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
ES = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )
lr_monitor = LearningRateMonitor(logging_interval='epoch')

GPU = DeviceStatsMonitor()
trainer = pl.Trainer(accelerator="gpu",
                     # profiler = 'advanced',
                     # strategy='deepspeed_stage_2',
                     precision = 'bf16',
                     max_epochs=20,
                    devices='auto',
                    accumulate_grad_batches=ACCUM_BATCH // TRAIN_BATCH,
                    limit_train_batches=0.2,
                    limit_val_batches=0.2,
                    # detect_anomaly=True,
                    callbacks=[checkpoint_callback,lr_monitor, GPU]
                    )
trainer.fit(lit, train_dataloader, test_dataloader)


