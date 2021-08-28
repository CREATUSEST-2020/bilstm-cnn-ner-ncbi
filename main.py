import argparse
from train import Trainer
from test import Inference
from utils import init_logger, build_vocab
from data_loader import load_examples


def main(args):
    init_logger()
    build_vocab(args)
    if args.train:
        train_dataset = load_examples(args, mode="train")
        dev_dataset = load_examples(args, mode="dev")
        test_dataset = None
        trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)
        trainer.train()
    if args.eval:
        test_dataset = load_examples(args, mode="test")
        inference = Inference(args, test_dataset)
        inference.load_model()
        inference.evaluate("test","test")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path for saving model")
    parser.add_argument("--embed_dir", default="./embeddings", type=str, help="Path for pretrained word vector")
    parser.add_argument("--vocab_dir", default="./vocab", type=str)
    parser.add_argument("--pred_dir", default="./preds", type=str, help="The prediction file dir")

    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--dev_file", default="devel.tsv", type=str, help="Test file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file")
    parser.add_argument("--embed_file", default="glove.6B.300d.txt", type=str, help="Pretrained word vector file")

    parser.add_argument("--max_seq_len", default=124, type=int, help="Max sentence length")
    parser.add_argument("--max_word_len", default=20, type=int, help="Max word length")
    parser.add_argument("--word_vocab_size", default=100000, type=int, help="Maximum size of word vocabulary")
    parser.add_argument("--char_vocab_size", default=1000, type=int, help="Maximum size of character vocabulary")

    parser.add_argument("--word_emb_dim", default=300, type=int, help="Word embedding size")
    parser.add_argument("--char_emb_dim", default=30, type=int, help="Character embedding size")
    parser.add_argument("--additional_emb_dim", default=50, type=int, help="Additional embedding size")
    parser.add_argument("--final_char_dim", default=50, type=int, help="Dimension of character cnn output")
    parser.add_argument("--hidden_dim", default=350, type=int, help="Dimension of BiLSTM output")

    parser.add_argument("--kernel_lst", default="2,3,4", type=str, help="kernel size for character cnn")
    parser.add_argument("--num_filters", default=32, type=int, help=" Number of filters for character cnn")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size for evaluation")
    parser.add_argument("--learning_rate", default=0.005, type=float, help="The initial learning rate")
    parser.add_argument("--num_train_epochs", default=15, type=float, help="Total number of training epochs to perform.")
    parser.add_argument('--save_steps', type=int, default=120, help="Save checkpoint every X updates steps.")

    parser.add_argument("--train", action="store_true", help="run training.")
    parser.add_argument("--eval", action="store_true", help="run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")


    args = parser.parse_args()

    main(args)
