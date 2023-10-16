 
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
os.environ['TRANSFORMERS_CACHE'] = './huggingface'

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
ROOT_FOLDER = "./data"

peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1
)


 
p_model = get_peft_model(model.encoder, peft_config)
p_model.print_trainable_parameters()
del model

collect()


prompt_train = pd.read_csv(os.path.join(ROOT_FOLDER, "prompts_train.csv"))
summaries_train = pd.read_csv(os.path.join(ROOT_FOLDER,"summaries_train.csv"))
combined = prompt_train.merge(summaries_train, on='prompt_id')

 
del prompt_train
del summaries_train

 
def map_prompts(dataframe):
    return f"""Topic: {dataframe['prompt_title']}\n
    Given the question : {dataframe['prompt_question']}\n
    Find the similarrity between the content text: {dataframe['prompt_text']}\n 
    and a summary written by a student: \n {dataframe['text']}
    """

 
combined['mapped'] = combined.apply(map_prompts, axis=1)

# OLD MODEL
"""
class LitModel(pl.LightningModule):

    def __init__(self, base_model, weight_decay = 1e4, lr= 1e4,lr_backbone=1e6):
        super().__init__()
        self.base_model = base_model
        self.lr_backbone = lr_backbone
        self.lr = lr
        self.weight_decay = weight_decay
        self.attention_pooling = AttentionPooling(self.base_model.config.d_model)
        self.linear = nn.Linear(self.base_model.config.d_model,2)

    def forward(self,input_ids, attention_mask, **kwargs):
        output = self.base_model(input_ids = input_ids, attention_mask=attention_mask)
        x = output.last_hidden_state
        x = self.attention_pooling(x, attention_mask.bool())
        x = self.linear(x)
        x = F.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx=None):
        y = batch.pop('labels')
        y_hat = self.forward(**batch)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx=None):
        y = batch.pop('labels')
        y_hat = self.forward(**batch)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
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
        optimizer = torch.optim.Adam(
            param_dicts
        )
        return optimizer
    
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_vector = nn.Parameter(torch.empty(self.hidden_dim))
        nn.init.normal_(self.attention_vector.data, std=0.02)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states, mask):
        attention_logits = hidden_states.matmul(self.attention_vector)
        attention_logits = attention_logits.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(attention_logits, dim=-1)
        pooled_representation = torch.sum(attention_weights.unsqueeze(-1) * hidden_states, dim=1)
        return self.layer_norm(pooled_representation)

"""
class LitModel(pl.LightningModule):

    def __init__(self, base_model, weight_decay = 1e-4, lr= 1e-3,lr_backbone=1e-6):
        super().__init__()
        self.base_model = base_model
        self.lr_backbone = lr_backbone
        self.lr = lr
        self.weight_decay = weight_decay
        self.attention_pooling = AttentionPooling(self.base_model.config.d_model)
        # self.linear = nn.Linear(2,1)
        self.linear = nn.Linear((self.base_model.config.d_model-1),2)
        self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2)
    def forward(self,batch,**kwargs):
        input_text = batch[0]
        output_text = batch[1]
        input_state = self.base_model(**input_text).last_hidden_state
        output_state = self.base_model(**output_text).last_hidden_state
        input_state = self.attention_pooling(input_state, input_text.attention_mask.bool())
        output_state = self.attention_pooling(output_state, output_text.attention_mask.bool())
        x = torch.stack([input_state, output_state], dim=2)
        x = self.conv(x.permute(0,2,1))
        # x = F.relu(x)
        x = x.squeeze(1)
        x = self.linear(x)
        x = F.sigmoid(x)
        del input_text, output_text, input_state, output_state
        return x
        

    def training_step(self, batch, batch_idx=None):
        y = batch.pop()
        y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss.item() , on_step=True, on_epoch=True, prog_bar=True)
        del y_hat, y
        return loss
    
    def validation_step(self, batch, batch_idx=None):
        y = batch.pop()
        y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss.item() , on_step=True, on_epoch=True, prog_bar=True)
        del y_hat, y
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "base_model" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if ("attention_pooling" in n or "linear" in n or "conv" in n) and p.requires_grad
                ],
                "lr": self.lr,  # assuming self.lr > self.lr_backbone
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, weight_decay=self.weight_decay
        )
        return optimizer
    
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_vector = nn.Parameter(torch.empty(self.hidden_dim))
        nn.init.uniform_(self.attention_vector.data)
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim)

    def forward(self, hidden_states, mask):
        attention_logits = hidden_states.matmul(self.attention_vector)
        attention_logits = attention_logits.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(attention_logits, dim=-1)

        pooled_representation = torch.sum(attention_weights.unsqueeze(-1) * hidden_states, dim=1)
        del attention_logits, attention_weights
        return self.batch_norm(pooled_representation)
    
class SummaryDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx,-4:].values
    
import torch
def collate_fn(batch):
    input_text = [x[3] for x in batch]
    text = [x[0] for x in batch]
    content = [x[1] for x in batch]
    wording = [x[2] for x in batch]
    tokenized_input = tokenizer(input_text, padding=True, return_tensors = 'pt')
    tokenized_output = tokenizer(text, padding=True, return_tensors = 'pt')
    labels = torch.tensor(list(zip(content, wording))).float()
    return [tokenized_input, tokenized_output, labels]

lit = LitModel(p_model)

 
train_df = combined.sample(frac= 0.8)
test_df = combined.drop(train_df.index)

 
from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler()
train_df[['content','wording']] = scl.fit_transform(train_df[['content','wording']])
test_df[['content','wording']] = scl.transform(test_df[['content','wording']])

 
train_dataset = SummaryDataset(train_df)
test_dataset =SummaryDataset(test_df)
train_dataloader= DataLoader(train_dataset, batch_size= 2, shuffle=True, collate_fn=collate_fn, num_workers=0)
test_dataloader= DataLoader(test_dataset, batch_size= 4,shuffle=True,collate_fn=collate_fn, num_workers=0)

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateFinder, ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{val_loss:.2f}",
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


trainer = pl.Trainer(accelerator="gpu",
                     profiler = 'advanced',
                     # strategy='deepspeed_stage_2',
                     precision = 'bf16',
                     max_epochs=5,
                    devices='auto',
                    accumulate_grad_batches=8,
                    limit_train_batches=0.2,
                    limit_val_batches=0.2,
                    # detect_anomaly=True,
                    callbacks=[checkpoint_callback,ES]
                    )
trainer.fit(lit, train_dataloader, test_dataloader)


