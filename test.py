import os
import shutil
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam

from data_loader import load_word_matrix
from utils import set_seed, load_vocab, compute_metrics, show_report, get_labels, get_test_texts
from model import BiLSTM_CNN

logger = logging.getLogger(__name__)


class Inference(object):
    def __init__(self, args, test_dataset=None):
        self.args = args
        self.test_dataset = test_dataset

        self.label_lst = get_labels(args)
        self.num_labels = len(self.label_lst)

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = 0

        self.word_vocab, self.char_vocab, _, _ = load_vocab(args)
        self.pretrained_word_matrix = None
        
        self.pretrained_word_matrix = load_word_matrix(args, self.word_vocab)

        self.model = BiLSTM_CNN(args, self.pretrained_word_matrix)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)
        self.eval = args.eval
        self.test_texts = None
        if self.eval:
            self.test_texts = get_test_texts(args)
            # Empty the original prediction files
            if os.path.exists(args.pred_dir):
                shutil.rmtree(args.pred_dir)
                

    def evaluate(self, mode, step):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'eval':
            dataset = self.dev_dataset
        elif mode == 'train':
            dataset = self.train_dataset
        else:
            raise Exception("Only train, dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'word_ids': batch[0],
                          'char_ids': batch[1],
                          'mask': batch[2],
                          'additional' : batch[3],
                          'label_ids': batch[4]}
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if preds is None:
                batch_size = logits.size(0)
                _, arg_max = torch.max(logits, dim=2)
                preds = []
                for i in range(batch_size):
                    preds.append(arg_max[i].cpu().data.numpy())
                out_label_ids = inputs["label_ids"].detach().cpu().numpy()
            else:
                out_label_ids = np.append(out_label_ids, inputs["label_ids"].detach().cpu().numpy(), axis=0)
                batch_size = logits.size(0)
                _, arg_max = torch.max(logits, dim=2)
                for i in range(batch_size):
                    preds.append(arg_max[i].cpu().data.numpy())
                

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]
        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])

        if self.eval:
            if not os.path.exists(self.args.pred_dir):
                os.mkdir(self.args.pred_dir)
            with open(os.path.join(self.args.pred_dir, "pred_{}.txt".format(step)), "w", encoding="utf-8") as f:
                for text, true_label, pred_label in zip(self.test_texts, out_label_list, preds_list):
                    for t, tl, pl in zip(text.split(), true_label, pred_label):
                        f.write("{} {} {}\n".format(t, tl, pl))
                    f.write("\n")
        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("\n" + show_report(out_label_list, preds_list))  # Get the report for each tag result

        return results


    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.args = torch.load(os.path.join(self.args.model_dir, 'args.pt'))
            logger.info("***** Args loaded *****")
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_dir, 'model.pt')))
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
