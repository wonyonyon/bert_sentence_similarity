# -*- coding: utf-8 -*-
# -------------------------------------

import torch.nn as nn
import torch
import os
from torch.utils.data import DataLoader,RandomSampler,TensorDataset
from transformers import BertConfig, BertModel,BertForSequenceClassification,BertTokenizer
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.data_helper import QuerySimilarityProcessor, load_data
# import pandas as pd
# pd.options.display.max_columns = None


processor = QuerySimilarityProcessor()
label_list = processor.get_labels()

num_labels = len(label_list)
config = BertConfig.from_pretrained("../data/bert-base-chinese/config.json", num_labels=num_labels)
model = BertForSequenceClassification.from_pretrained("../data/bert-base-chinese/")
tokenizer = BertTokenizer.from_pretrained('../data/bert-base-chinese/vocab.txt')

train_dataset = load_data("../data/ATEC/", processor, tokenizer)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32)

valid_dataset = load_data("../data/ATEC/", processor, tokenizer,evaluate=True)
valid_sampler = RandomSampler(train_dataset)
valid_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32)

config_class, model_class, tokenizer_class = [BertConfig, BertForSequenceClassification, BertTokenizer]
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=0)


def train():
    model.train()
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        batch_start = time.time()
        batch = tuple(t.to("cpu") for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels': batch[3]}
        optimizer.zero_grad()
        outputs = model(**inputs)
        # loss = outputs[0]  # models outputs are always tuple in transformers (see doc)
        loss, logits = outputs[:2]
        # print(outputs)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()

        correct_preds += correct_predictions(logits, batch[3])

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (step + 1),
                    running_loss / (step + 1))
        epoch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(train_dataloader)
    epoch_accuracy = correct_preds / len(train_dataloader.dataset)

    return epoch_time, epoch_loss, epoch_accuracy


def validate():
    model.eval()
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    result_preds = []

    epoch_iterator = tqdm(valid_dataloader, desc="Iteration")
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            batch_start = time.time()
            batch = tuple(t.to("cpu") for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            optimizer.zero_grad()
            outputs = model(**inputs)
            # loss = outputs[0]  # models outputs are always tuple in transformers (see doc)
            loss, logits = outputs[:2]
            # print(outputs)
            result_preds.append(np.array(logits.cpu()))
            running_loss += loss.item()
            running_accuracy += correct_predictions(logits, batch[3])
            running_loss += loss.item()

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(valid_dataloader)
    epoch_accuracy = running_accuracy / len(valid_dataloader.dataset)

    return epoch_time, epoch_loss, epoch_accuracy, np.vstack(result_preds)


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    checkpoints of a models.
    Args:
        output_probabilities: A tensor of probabilities for different checkpoints
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


epochs = 10
epochs_count = []
train_losses = []
valid_losses = []
best_score = 0.0
patience_counter = 0
patience = 5
save_flag = True
target_dir = '../data/checkpoints'
for epoch in range(1, epochs+1):
    # epochs_count.append(epoch)

    print("* Training epoch {}:".format(epoch))
    epoch_time, epoch_loss, epoch_accuracy = train()

    train_losses.append(epoch_loss)
    print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
          .format(epoch_time, epoch_loss, (epoch_accuracy*100)))
    print("* Validation for epoch {}:".format(epoch))
    epoch_time, epoch_loss, epoch_accuracy, prob = validate()

    valid_losses.append(epoch_loss)
    print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
          .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))

    if epoch_accuracy < best_score:
        patience_counter += 1
    else:
        best_score = epoch_accuracy
        patience_counter = 0
        #         Save the best models. The optimizer is not saved to avoid having
        #         a checkpoint file that is too heavy to be shared. To resume
        #         training from the best models, use the 'esim_*.pth.tar'
        #         checkpoints instead.
        if save_flag:
            torch.save({"epoch": epoch,
                        "models": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "best.pth.tar"))

        # Save the models at each epoch.
    if save_flag:
        torch.save({"epoch": epoch,
                    "models": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                   os.path.join(target_dir, "esim_{}.pth.tar".format(epoch)))

    if patience_counter >= patience:
        print("-> Early stopping: patience limit reached, stopping...")
        break


plt.figure()
plt.plot(epochs_count, train_losses, "-r")
plt.plot(epochs_count, valid_losses, "-b")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["Training loss", "Validation loss"])
plt.title("Cross entropy loss")
plt.show()