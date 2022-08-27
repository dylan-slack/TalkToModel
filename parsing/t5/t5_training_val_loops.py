"""Training epoch

Modified from: https://github.com/Shivanandroy/T5-Finetuning-PyTorch
"""
import torch


def validate(tokenizer, model, device, loader, debug=False):
    """Validation loop"""
    model.eval()
    losses = 0.
    items = 0.
    predictions, ground_truth = [], []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data["target_ids"].to(device, dtype=torch.long)
            lm_labels = y.clone().detach()
            lm_labels[y == tokenizer.pad_token_id] = -100
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                labels=lm_labels,
            )

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                early_stopping=True,
            )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

            predictions.extend(preds)
            ground_truth.extend(target)

            loss = outputs[0].detach()

            losses += loss
            items += ids.shape[0]

            if _ % 50 == 0:
                print(f'Validated {_}', flush=True)

            if _ > 3 and debug:
                break

    avg_loss = losses / items
    return avg_loss, predictions, ground_truth


def train(epoch, tokenizer, model, device, loader, optimizer, debug=False):
    """One epoch"""

    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        lm_labels = y.clone().detach()
        lm_labels[y == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 100 == 0:
            print(f"epoch {str(epoch)} | step {_} | loss: {str(loss)}", flush=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # break early for debugging
        if _ == 2 and debug:
            break
