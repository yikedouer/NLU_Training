import glob
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
# from lion_pytorch import Lion
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import transformers
from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizerFast, AutoTokenizer,
                          get_cosine_with_hard_restarts_schedule_with_warmup, 
                          get_polynomial_decay_schedule_with_warmup,
                          get_linear_schedule_with_warmup, logging)

from callback.adversarial import FGM
from callback.ema import EMA
from metric.slot_metrics import SlotMetrics
from metric.intent_metrics import IntentMetrics
from models.bert_joint_nlu import BertJointNlu
from models.bert_intent_classifier import BertIntentClassifier
from processors.task_processor import collate_fn, convert_examples_to_features
from processors.task_processor import task_processors as processors
from processors.utils import get_entities
from tools.argparser import get_argparse
from tools.common import (init_logger, logger, pretty_config_info,
                          seed_everything, seed_worker, save_pickle, load_pickle)
from tools.progressbar import ProgressBar
from configs.model_config import MODEL_CLASSES

logging.set_verbosity_error()


# MODEL_CLASSES = {
#     ## bert
#     'bert_nlu': (BertConfig, BertJointNlu, AutoTokenizer),
#     'bert_intent': (BertConfig, BertIntentClassifier, AutoTokenizer),
#     # 'albert': (AlbertConfig, AlbertCrfForNer, CNerTokenizer)
# }


def train(args, train_dataset, model, tokenizer, ema=None):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn, num_workers=4, worker_init_fn=seed_worker)
    if args.max_steps > 0:
        total_step = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        total_step = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    linear_param_optimizer = list(model.intent_classifier.named_parameters())

    optimizer_grouped_parameters = [
        # bert params
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},

        # linear params
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.liner_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.liner_learning_rate},
    ]

    if args.freeze_pretrain:
        for name, param in model.named_parameters():
            if "bert" in name:
                param.requires_grad = False

    args.warmup_steps = int(total_step * args.warmup_proportion)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_step, num_cycles=args.num_cycles)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    # can be replaced with pytorch autocast
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = "
                f"{args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {total_step}")

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {global_step}")
        logger.info(f"  Will skip the first {steps_trained_in_current_epoch} steps in the first epoch")

    train_loss = 0.0
    
    fgm = FGM(model, emb_name=args.adv_name, epsilon=args.adv_epsilon) if args.do_adv else None
    model.zero_grad()
    # Added here for reproducibility
    seed_everything(args.seed)
    pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=int(args.num_train_epochs))
    args.save_steps = len(train_dataloader) if args.save_steps == -1 else args.save_steps
    args.logging_steps = len(train_dataloader) if args.logging_steps == -1 else args.logging_steps
    best_intent_f1 = 0.0
    no_gain_counter = 0
    
    for epoch in range(int(args.num_train_epochs)):
        if no_gain_counter >= 50:
            return global_step, train_loss / global_step
        pbar.reset()
        pbar.epoch_start(current_epoch=epoch)
        if args.local_rank != -1:
            train_dataloader.sampler.set_epoch(epoch) 
        # 每个epoch中各个step的loss等信息存储起来，在epoch结束后统一上报至wandb，减少GPU等待时间，提高利用率
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "intent_labels_ids": batch[4],
                "token_type_ids": batch[2] if args.model_type in ["bert_intent"] else None
            }
            if args.speaker_segment:
                inputs["token_type_ids"] = batch[6]
            nlu_loss, nlu_logits = model(**inputs)
            loss, kl_loss, intent_loss = nlu_loss.total_loss, nlu_loss.kl_loss_intent, nlu_loss.intent_loss
            if args.n_gpu > 1:
                loss = loss.mean()
                kl_loss = kl_loss.mean()
                intent_loss = intent_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                kl_loss = kl_loss / args.gradient_accumulation_steps
                intent_loss = intent_loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if args.do_adv:
                fgm.attack()
                loss_adv, _ = model(**inputs)
                loss_adv = loss_adv.total_loss
                if args.n_gpu > 1:
                    loss_adv = loss_adv.mean()
                loss_adv.backward()
                fgm.restore()

            pbar(step, {'loss': loss.item(), "intent loss": intent_loss.item(), "kl loss": kl_loss.item()})
            train_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # Update learning rate schedule
                optimizer.step()
                if args.do_ema:
                    ema.update()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0]:# and global_step % 50 == 0:
                    # Only logging to wandb when single GPU, or DP-multi-GPU or the 1st process in DDP
                    # 可以每个step的指标暂存在GPU，每个epoch结束后统一wandb.log, 减少cpu、gpu的频繁数据交换
                    lrs = scheduler.get_last_lr()
                    wandb.log({
                        "train/total_loss": loss.item(),
                        "train/intent_loss": intent_loss.item(),
                        "train/kl_loss": kl_loss.item(),
                        "train/learning_rate": lrs[0],
                        "train/linear_learning_rate": lrs[-1],
                    }, step=global_step)
                if args.local_rank in [-1, 0] and args.evaluate_during_training and global_step % args.logging_steps == 0:
                    eval_results = evaluate(args, model, tokenizer, prefix=global_step, ema=ema)
                    # evaluate(args, model, tokenizer, prefix=global_step, ema=ema, dev_prefix="live")
                    # evaluate(args, model, tokenizer, prefix=global_step, ema=ema, dev_prefix="short_video")
                    intent_f1 = float(eval_results["intent_micro_f1"])
                    
                    if intent_f1 > best_intent_f1:
                        logger.info(f"best f1 now update: ")
                        logger.info(f"intent f1: {best_intent_f1} -> {intent_f1}")
                        output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                        # Take care of distributed/parallel training
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        model_to_save.save_pretrained(output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        tokenizer.save_vocabulary(output_dir)
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                    best_intent_f1 = max(intent_f1, best_intent_f1)

                    wandb.log({
                        "eval/intent_micro_f1": intent_f1,
                        "eval/eval_loss": float(eval_results["loss"]),
                    }, step=global_step)

                    # 没有进一步变好, 计数器加1
                    if intent_f1 < best_intent_f1:
                        no_gain_counter += 1
                    # FIXME: remove these saving code below
                    output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step+1}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    # Take care of distributed/parallel training
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    
        logger.info("\n")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, train_loss / global_step

@torch.no_grad()
def evaluate(args, model, tokenizer, prefix="", step=None, ema=None, dev_prefix=None):
    if ema is not None:
        ema.apply_shadow()
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev', prefix=dev_prefix)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    # FIXME: debug
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn, num_workers=4, worker_init_fn=seed_worker)
    # Eval!
    logger.info(f"***** Running evaluation {prefix} *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    if isinstance(model, nn.DataParallel):
        model = model.module
    all_intent_labels_ids, all_intent_preds = [], []
    logging_intent_preds, logging_intent_labels = [], []
    sentence_list, logging_data = [], []
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "intent_labels_ids": batch[4],
            "token_type_ids": batch[2] if args.model_type in ["bert_intent"] else None,
            "is_train": False
        }
        nlu_loss, nlu_logits = model(**inputs)
        tmp_eval_loss = nlu_loss.total_loss
        intent_logits = nlu_logits.intent_logits
        # TODO: 增加参数区分多分类和多标签
        if args.multi_label:
            intent_logits = torch.sigmoid(intent_logits)
        else:
            intent_logits = intent_logits.softmax(dim=-1)
        # print(f"inputs['attention_mask'] shape: {inputs['attention_mask'].shape}")
        
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        intent_labels_ids = batch[4].cpu().numpy().tolist()
        
        if args.multi_label:
            intent_logits = intent_logits.cpu().numpy().tolist()
            intent_pred_ids = [[1 if logit > 0.5 else 0 for logit in logits] for logits in intent_logits]
        else:
            intent_logits = intent_logits.cpu().numpy()
            intent_pred_ids = [logits.argmax() for logits in intent_logits]

        all_intent_labels_ids.extend(intent_labels_ids)
        all_intent_preds.extend(intent_pred_ids)
        for label_ids, pred_ids in zip(intent_labels_ids, intent_pred_ids):
            if args.multi_label:
                label = [args.intent_id2label[idx] for idx, label_idx in enumerate(label_ids) if label_idx == 1]
                pred = [args.intent_id2label[idx] for idx, pred_idx in enumerate(pred_ids) if pred_idx == 1]
            else:
                label = args.intent_id2label[label_ids]
                pred = args.intent_id2label[pred_ids]
            logging_intent_preds.append(pred)
            logging_intent_labels.append(label)

        pbar(step)

    # logging intent eval metrics
    intent_metric = IntentMetrics(args.intent_label_list)
    micro_info, macro_info, class_info = intent_metric.compute(all_intent_labels_ids, all_intent_preds)
    if not micro_info:
        micro_info = macro_info
    inent_micro_f1 = micro_info["f1-score"]
    logger.info("")
    logger.info(f"Intent Report for Step {prefix} {dev_prefix}: ")
    logger.info("intent\tprecision\trecall\tf1-score\tsupport")
    for k, v in class_info.items():
        logger.info(f'{k}\t{v["precision"]}\t{v["recall"]}\t{v["f1-score"]}\t{v["support"]}')
    logger.info(f"micro\t{micro_info['precision']}\t{micro_info['recall']}\t{micro_info['f1-score']}\t{micro_info['support']}")
    logger.info("\n")


    eval_loss = eval_loss / nb_eval_steps
    results = {"loss": eval_loss,  "intent_micro_f1": inent_micro_f1}
    for idx, sentence in enumerate(sentence_list):
        sentence = sentence.split("[SEP]")
        context, query = sentence[0], sentence[1]
        line = f"{context}\t{query}\t{logging_intent_labels[idx]}\t{logging_intent_preds[idx]}"
        logging_data.append(line)
    # save all eval result to disk
    result_file = f"eval_result_{prefix}.csv"
    csv_output_path = os.path.join(args.output_dir, result_file)

    with open(csv_output_path, "w") as fp:
        fp.write("context\tquery\tintent_label\tintent_pred\n")
        fp.write("\n".join(logging_data))
    if ema is not None:
        ema.restore()
    return results

@torch.no_grad()
def export(args, model, tokenizer, type="trace"):
    try:
        import lego
        import laplace as la
    except ImportError:
        raise ImportError("Please install lego and laplace from https://bytedance.feishu.cn/docs/doccnyl3hTxS4e1GjSHtbkI0Vvg.")
    saved_dir = f"{args.output_dir}/saved_models/"
    if not os.path.exists(saved_dir) and args.local_rank in [-1, 0]:
        os.makedirs(saved_dir)

    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=32,
                                 collate_fn=collate_fn, num_workers=1, worker_init_fn=seed_worker)
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.to(args.device)
    model.eval()
    batch = next(iter(eval_dataloader))
    batch = tuple(t.to(args.device) for t in batch)
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],
        "is_train": torch.tensor(False),
        "export": torch.tensor(True)
    }
    if type == "trace":
        logger.info("Use trace")
        jit_model = torch.jit.trace(model.half() ,(inputs['input_ids'].cuda(), inputs['attention_mask'].cuda(), inputs['token_type_ids'].cuda()))
        logits = jit_model(inputs['input_ids'].cuda(),inputs['attention_mask'].cuda(),inputs['token_type_ids'].cuda())
        logger.info(f"jit_model.graph: \n{jit_model}")
        jit_model.save(os.path.join(saved_dir, "jit_model.pt"))
        lego.torch_load_lego_library()
        logger.info("load lego opt done.")
        lego_input = (inputs['input_ids'].cuda(), inputs['attention_mask'].cuda(), inputs['token_type_ids'].cuda())
        lego.set_lego_thresold(0.5)
        lego_model = lego.optimize(os.path.join(saved_dir, "jit_model.pt"), lego_input, remove_padding=False)
        logger.info("optimize done")
        if not os.path.exists(os.path.join(saved_dir, "lego_model")):
            os.makedirs(os.path.join(saved_dir, "lego_model"))
        lego_model.save(os.path.join(saved_dir, "lego_model", "model.pt"))
        logger.info("save lego model done.")
        logger.info("reload lego model: ")
        lego_model2 = torch.jit.load(os.path.join(saved_dir, "lego_model", "model.pt"))
        original_model = torch.jit.load(os.path.join(saved_dir, "jit_model.pt"))
        lego.perf_model(original_model, lego_model, lego_input)
        logits = lego_model2(inputs['input_ids'].cuda(),inputs['attention_mask'].cuda(),inputs['token_type_ids'].cuda())
        logger.info(f"logits[1] shape: {logits[1].shape}")

    else:
        lego.torch_load_lego_library()
        logger.info("Use script, and not implemented")
        slot_pred_list, slot_true_list = [], []
        all_intent_labels_ids, all_intent_preds = [], []
        logging_intent_preds, logging_intent_labels = [], []
        logging_pred_entities, logging_label_entities = [], []
        sentence_list, logging_data = [], []
        slot_metric = SlotMetrics(args.slot_id2label)
        lego_model = lego_model = torch.jit.load(os.path.join(saved_dir, "lego_model_629", "model.pt"))
        lego_model.eval()
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "intent_labels_ids": batch[4],
                "slot_labels_ids": batch[5],
                "token_type_ids": batch[2] if args.model_type in ["bert_intent"] else None,
                "is_train": False
            }
            lego_input = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            intent_logits, slot_logits = model(**lego_input)
            # print(f"inputs['attention_mask'] shape: {inputs['attention_mask'].shape}")
            slot_mask = inputs['attention_mask'][:, 102:-1]
            
            tags = slot_logits.cpu().numpy()

            slot_labels_ids = inputs['slot_labels_ids']
            slot_labels_ids = slot_labels_ids if args.slot_decoder == "global_pointer" else slot_mask * slot_labels_ids
            slot_labels_ids = slot_labels_ids.cpu().numpy()
            for i, label in enumerate(slot_labels_ids):
                truth, pred = [], []
                # global_pointer的预测矩阵过大，不保存到预测文件
                if args.slot_decoder != "global_pointer":
                    truth = [args.slot_id2label[m] if args.slot_id2label[m] != "X" else "O" for j, m in enumerate(label)]
                    pred = [args.slot_id2label[m] if args.slot_id2label[m] != "X" else "O" for j, m in enumerate(tags[i])]

                slot_true_list.append(truth)
                slot_pred_list.append(pred)
                true_entities = get_entities(label, args.slot_id2label, markup=args.markup)
                pred_entities = get_entities(tags[i], args.slot_id2label, markup=args.markup)
                logging_label_entities.append(true_entities)
                logging_pred_entities.append(pred_entities)

                slot_metric.update(true_entities=true_entities, pre_entities=pred_entities)
                sentence = "".join(tokenizer.convert_ids_to_tokens(inputs["input_ids"][i]))
                sentence_list.append(sentence.replace("[PAD]", "").replace("[CLS]", ""))
                # print(f"sentence: {sentence}")
                # print(f"true_entities: {true_entities}")
                # print(f"pred_entities: {pred_entities}")


        # logging slot eval metrics
        total_info, slot_info = slot_metric.result()
        slot_micro_f1 = total_info["f1-score"]
        logger.info("slot\tprecision\trecall\tf1-score\tsupport")
        for k, v in slot_info.items():
            logger.info(f'{k}\t{v["precision"]}\t{v["recall"]}\t{v["f1"]}\t{v["support"]}')
        logger.info(f"micro\t{total_info['precision']}\t{total_info['recall']}\t{total_info['f1-score']}\t{total_info['support']}")

    # warmup data
    # feed_dict = {
    #     "input_ids": la.make_tensor_proto(inputs["input_ids"].cpu().int()).SerializeToString(),
    #     "attention_mask": la.make_tensor_proto(inputs["attention_mask"].cpu().int()).SerializeToString(),
    #     "token_type_ids": la.make_tensor_proto(inputs["token_type_ids"].cpu().int()).SerializeToString(),
    # }
    
    # warmup_data_path=os.path.join(saved_dir, "lego_model", "assets.extra")
    # if not os.path.exists(warmup_data_path):
    #     os.mkdir(warmup_data_path)
    # la.Model.make_inference_warmup_data(os.path.join(warmup_data_path, "tf_serving_warmup_requests"), feed_dict)


def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev')
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)
    results = []
    output_predict_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")

    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None, 'input_lens': batch[4]}
            if args.model_type != "distilbert":
                # bert need using segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert_intent"] else None)
            outputs = model(**inputs)
            logits = outputs[0]
            tags = model.crf.decode(logits, inputs['attention_mask'].byte())
            tags = tags.squeeze(0).cpu().numpy().tolist()
        # [CLS]XXXX[SEP]
        preds = tags[0][1:-1]
        label_entities = get_entities(preds, args.id2label, args.markup)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join([args.id2label[x] for x in preds])
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step)
    logger.info("\n")
    with open(output_predict_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')

def get_mem_info():
    import psutil
    mem = psutil.virtual_memory()
    total = float(mem.total) / 1024 / 1024 / 1024
    used = float(mem.used) / 1024 / 1024 / 1024
    free = float(mem.free) / 1024 / 1024 / 1024

    logger.info(f"total memory: {total:.3f}")
    logger.info(f"used memory: {used:.3f}")
    logger.info(f"free memory: {free:.3f}")




def load_and_cache_examples(args, task, tokenizer, data_type='train', prefix=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache
        torch.distributed.barrier()
    processor = processors[task](task_name=task, slot_markup=args.markup, model_type=args.model_type)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_-{}_{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length),
        str(task),
        str(prefix)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = load_pickle(cached_features_file)
        # features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        intent_label_list, _ = processor.get_labels(slot_markup=args.markup)
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir, in_context=args.in_context, prefix=prefix)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir, in_context=args.in_context, prefix=prefix)
        else:
            examples = processor.get_test_examples(args.data_dir, in_context=args.in_context, prefix=prefix)
        features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
                                                intent_label_list=intent_label_list,
                                                max_seq_length=args.train_max_seq_length if data_type == 'train' \
                                                    else args.eval_max_seq_length,
                                                args=args, in_context=args.in_context)
        if args.local_rank in [-1, 0]:
            logger.info(f"Saving features into cached file {cached_features_file}")
            save_pickle(features, cached_features_file)

            logger.info("save feature done.")
    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache
        torch.distributed.barrier()
    # Convert to Tensors and build dataset
    all_input_ids = torch.from_numpy(np.array([f.input_ids for f in features])).type(torch.int32)
    all_input_mask = torch.from_numpy(np.array([f.input_mask for f in features])).type(torch.int32)
    all_segment_ids = torch.from_numpy(np.array([f.segment_ids for f in features])).type(torch.int32)
    all_intent_label_ids = torch.from_numpy(np.array([f.intent_label_ids for f in features])).type(torch.int32)
    all_slot_label_ids = torch.from_numpy(np.array([f.slot_label_ids for f in features])).type(torch.int32)
    all_speaker_segment_ids = torch.from_numpy(np.array([f.speaker_flag for f in features])).type(torch.int32)
    # torch.from_numpy(np.array([f.slot_label_ids for f in features])).type(torch.int32)
    
    all_lens = torch.from_numpy(np.array([f.input_len for f in features])).type(torch.int32)
    
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens,
                            all_intent_label_ids, all_slot_label_ids, all_speaker_segment_ids)
    del features
    return dataset


def main():
    args = get_argparse().parse_args()
    # Attention: sapce is illegal in wandb name
    if args.local_rank in [-1, 0] and args.do_train:
        name = args.wandb_desc.replace(" ", "_") if args.wandb_desc != "" else time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        wandb.init(project=args.task_name, name=name) 
        wandb.config.update(args)


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_dir = f"{args.output_dir}_{args.model_type}"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    # init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    log_file_name = f"{args.model_type}-{args.task_name}-{time_}.log"
    init_logger(log_file=os.path.join(args.output_dir, log_file_name))
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
        f"Process rank: {args.local_rank}, device: {device}, n_gpu: {args.n_gpu}, distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}")

    # Set seed
    seed_everything(args.seed)
    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name](task_name=args.task_name, model_type=args.model_type)
    intent_label_list, _ = processor.get_labels(slot_markup=args.markup)

    args.intent_id2label = {i: label for i, label in enumerate(intent_label_list)}
    args.intent_label2id = {label: i for i, label in enumerate(intent_label_list)}
    args.slot_label2id = None
    args.slot_id2label = None
    args.intent_label_list = intent_label_list # TODO

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None, )
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, use_fast=True, do_lower_case=False)
    model = model_class.from_pretrained(args.model_name_or_path, config=config, args=args,
                                        num_intent_labels=len(intent_label_list))
    if args.local_rank == 0:
        # Make sure only the first process will download model&vocab
        torch.distributed.barrier()
    # special_tokens_dict = {'additional_special_tokens': ["[USR]", "[SYS]", "[BIZ]", "[EDU]", "[MED]", "[HOME]"]}
    # ind_tokens = ['[食品饮料]', '[家居建材]', '[日化]', '[生活服务]', '[教育]', '[房地产]', '[金融业]', '[社会公共]', '[商务服务]', '[出行旅游]', '[汽车]', '[招商加盟]', '[医疗健康]']
    
    special_tokens_dict = {'additional_special_tokens': ['[USR]', '[SYS]', '[CAR]', '[HOME]', '[REALTY]', '[FINANCE]', '[EDU]', '[LIFE]', '[PHOTO]', '[BIZ]', '[INVEST]', '[CULTURE]', '[HEALTH]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    args_info = pretty_config_info(args)
    if args.local_rank in [-1, 0]:
        logger.info(f"Training/evaluation parameters:\n {args_info}")
    
    ema = None
    if args.do_ema:
        ema = EMA(model, 0.999)
        ema.register()

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        global_step, train_loss = train(args, train_dataset, model, tokenizer, ema=ema)
        logger.info(f" global_step = {global_step}, average loss = {train_loss}")

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.checkpoint == "all":
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True), key=lambda x: int(x.split("/")[-2].split("-")[-1]))
            )
            
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
        else:
            checkpoints = [f"{args.output_dir}/{args.checkpoint}"]
        # Reduce logging
        transformers.logging.set_verbosity_info()
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            if prefix == "":
                continue
            model = model_class.from_pretrained(checkpoint, config=config, args=args, num_intent_labels=len(intent_label_list))
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix, ema=ema)
            logger.info("\n")
            logger.info(result["intent_micro_f1"])
            
            results.update({int(global_step): result})

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write(f"{key}")
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        checkpoint = f"{args.output_dir}/{args.checkpoint}/"
        logger.info(f"Predict the following checkpoint: {checkpoint}")
        prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
        model = model_class.from_pretrained(checkpoint, config=config)
        model.to(args.device)
        predict(args, model, tokenizer, prefix=prefix)
    
    if args.do_export and args.local_rank in [-1, 0]:
        checkpoint = f"{args.output_dir}/{args.checkpoint}"
        # Reduce logging
        # logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)
        transformers.logging.set_verbosity_info()
        logger.info(f"Export the following checkpoints: {checkpoint}")
        global_step = checkpoint.split("-")[-1]
        model = model_class.from_pretrained(checkpoint, config=config, args=args, num_intent_labels=len(intent_label_list), do_export=True)
        export(args, model, tokenizer, type="trace")




if __name__ == "__main__":
    main()
