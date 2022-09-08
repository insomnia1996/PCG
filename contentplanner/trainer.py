# coding=utf-8
import sys
sys.path.append(r'../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random
import numpy as np
import time
import logging
import progressbar

import logging
logging.getLogger('transformers.generation_utils').disabled = True


def pred_model(args, model, data, cuda_available, device):
    dataset_batch_size = args.batch_size_per_gpu
    eval_step = len(data)
    model.eval()
    reference_list, prediction_list = [], []
    val_mle_loss, val_crf_loss = 0., 0.
    with torch.no_grad():
        p = progressbar.ProgressBar(eval_step)
        p.start()
        for idx, dev_iter in enumerate(data):
            p.update(idx)
            dev_batch_field, dev_batch_text, dev_batch_tgt = dev_iter['key'], dev_iter['value'],dev_iter['plan_id']
            if cuda_available:
                dev_batch_tgt = torch.tensor(dev_batch_tgt, dtype=torch.int64).to(device)
            one_reference_batch = model.parse_batch_output(dev_batch_tgt, dev_batch_field)
            reference_list += one_reference_batch
            one_prediction_batch = model.decode(dev_batch_field, dev_batch_text)
            prediction_list += one_prediction_batch
            one_val_mle_loss, one_val_crf_loss = model(dev_batch_field, dev_batch_text, dev_batch_tgt)
            val_mle_loss += one_val_mle_loss.item()
            val_crf_loss += one_val_crf_loss.item()
        assert len(reference_list) == len(prediction_list)
        p.finish()
    model.train()
    val_mle_loss /= eval_step
    val_crf_loss /= eval_step
    from utlis import measure_bleu_score
    bleu_score = measure_bleu_score(prediction_list, reference_list)
    return bleu_score, val_mle_loss, val_crf_loss

def eval_model(args, model, data, cuda_available, device):
    dataset_batch_size = args.batch_size_per_gpu
    eval_step = len(data)
    model.eval()
    reference_list, prediction_list = [], []
    val_mle_loss, val_crf_loss = 0., 0.
    with torch.no_grad():
        p = progressbar.ProgressBar(eval_step)
        p.start()
        for idx, dev_iter in enumerate(data):
            p.update(idx)
            dev_batch_field, dev_batch_text, dev_batch_tgt = dev_iter['key'], dev_iter['value'],dev_iter['plan_id']
            if cuda_available:
                dev_batch_tgt = torch.tensor(dev_batch_tgt, dtype=torch.int64).to(device)
            one_reference_batch = model.parse_batch_output(dev_batch_tgt, dev_batch_field)#parsing results
            reference_list += one_reference_batch
            one_prediction_batch = model.decode(dev_batch_field, dev_batch_text)#decode tokens, parsing results
            prediction_list += one_prediction_batch
            one_val_mle_loss, one_val_crf_loss = model(dev_batch_field, dev_batch_text, dev_batch_tgt)
            print("=======================")
            val_mle_loss += one_val_mle_loss.item()
            val_crf_loss += one_val_crf_loss.item()
        assert len(reference_list) == len(prediction_list)
        p.finish()
    model.train()
    val_mle_loss /= eval_step
    val_crf_loss /= eval_step
    from utlis import measure_bleu_score
    bleu_score = measure_bleu_score(prediction_list, reference_list)
    return bleu_score, val_mle_loss, val_crf_loss

def model_training(args, train_data, dev_data, model, total_steps, print_every, save_every, ckpt_save_path, cuda_available, device):
    import os
    if os.path.exists(ckpt_save_path):
        pass
    else: # recursively construct directory
        os.makedirs(ckpt_save_path, exist_ok=True)
    log_path = ckpt_save_path + '/log.txt'

    max_save_num = 1
    batch_size_per_gpu, gradient_accumulation_steps = \
    args.batch_size_per_gpu, args.gradient_accumulation_steps

    warmup_steps = int(0.1 * total_steps) # 10% of training steps are used for warmup
    print ('total training steps is {}, warmup steps is {}'.format(total_steps, warmup_steps))
    from transformers.optimization import AdamW, get_linear_schedule_with_warmup
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    optimizer.zero_grad()

    effective_batch_acm = 0
    all_batch_step = 1
    print_valid, save_valid = False, False
    train_mle_loss, train_crf_loss, max_val_bleu = 0., 0., 0.

    print ('--------------------------------------------------------------------------')
    print ('Start Training:')
    model.train()
    number_of_saves = 0

    while effective_batch_acm < total_steps:
        all_batch_step += 1
        if all_batch_step%len(train_data)==len(train_data)-1:
            train_data.reset()
        train_iter=next(train_data)

        train_batch_field, train_batch_text, train_batch_tgt = train_iter['key'], train_iter['value'],train_iter['plan_id']
        if cuda_available:
            train_batch_tgt = torch.tensor(train_batch_tgt, dtype=torch.int64).to(device)
        mle_loss, crf_loss = model(train_batch_field, train_batch_text, train_batch_tgt)

        loss = args.mle_loss_weight * mle_loss + crf_loss
        loss = loss.mean()
        loss.backward()

        train_mle_loss += mle_loss.item()
        train_crf_loss += crf_loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # parameter update
        if all_batch_step % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            effective_batch_acm += 1
            print_valid, save_valid = True, True

        # print intermediate result
        if effective_batch_acm % print_every == 0 and print_valid:
            denominator = (effective_batch_acm - (number_of_saves * save_every)) * gradient_accumulation_steps

            one_train_mle_loss = train_mle_loss / denominator
            one_train_crf_loss = train_crf_loss / denominator
            train_log_text = 'Training:At training steps {}, training MLE loss is {}, train CRF loss is {}'.format(effective_batch_acm, 
                one_train_mle_loss, one_train_crf_loss)
            print (train_log_text)
            with open(log_path, 'a', encoding='utf8') as logger:
                logger.writelines(train_log_text + '\n')
            print_valid = False

        # saving result
        if effective_batch_acm % save_every == 0 and save_valid:
            number_of_saves += 1

            save_valid = False

            one_train_mle_loss = train_mle_loss / (save_every * gradient_accumulation_steps)
            one_train_crf_loss = train_crf_loss / (save_every * gradient_accumulation_steps)

            model.eval()
            dev_data.reset()
            one_val_bleu_score, one_val_mle_loss, one_val_crf_loss = eval_model(args, model, dev_data, cuda_available, device)
            one_val_ppl = np.exp(one_val_mle_loss)
            one_val_ppl = round(one_val_ppl, 3)
            model.train()

            valid_log_text = 'Validation:At training steps {}, training MLE loss is {}, train CRF loss is {}, \
            validation MLE loss is {}, validation ppl is {}, validation CRF loss is {}, validation BLEU is {}'.format(effective_batch_acm, 
                one_train_mle_loss, one_train_crf_loss, one_val_mle_loss, one_val_ppl, one_val_crf_loss, one_val_bleu_score)
            print (valid_log_text)
            with open(log_path, 'a', encoding='utf8') as logger:
                logger.writelines(valid_log_text + '\n')

            train_mle_loss, train_crf_loss = 0., 0.

            if one_val_bleu_score > max_val_bleu:
                max_val_bleu = max(max_val_bleu, one_val_bleu_score)
                # in finetuning stage, we always save the model
                print ('Saving model...')
                save_name = 'training_step_{}_train_mle_loss_{}_train_crf_loss_{}_dev_mle_loss_{}_dev_ppl_{}_dev_crf_loss_{}_dev_bleu_{}'.format(effective_batch_acm,
                round(one_train_mle_loss,5), round(one_train_crf_loss,5), round(one_val_mle_loss,5), one_val_ppl, round(one_val_crf_loss,5), one_val_bleu_score)

                model_save_path = ckpt_save_path + '/' + save_name
                import os
                if os.path.exists(model_save_path):
                    pass
                else: # recursively construct directory
                    os.makedirs(model_save_path, exist_ok=True)
                if cuda_available and torch.cuda.device_count() > 1:
                    model.module.save_model(model_save_path)
                else:
                    model.save_model(model_save_path)
                print ('Model Saved!')

                # --------------------------------------------------------------------------------------------- #
                # removing extra checkpoints...
                import os
                from operator import itemgetter
                fileData = {}
                test_output_dir = ckpt_save_path
                for fname in os.listdir(test_output_dir):
                    if fname.startswith('training_step'):
                        fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                    else:
                        pass
                sortedFiles = sorted(fileData.items(), key=itemgetter(1))

                if len(sortedFiles) < max_save_num:
                    pass
                else:
                    delete = len(sortedFiles) - max_save_num
                    for x in range(0, delete):
                        one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                        os.system('rm -r ' + one_folder_name)
                print ('-----------------------------------')
                # --------------------------------------------------------------------------------------------- #
    return model

