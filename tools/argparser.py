import argparse


def get_argparse():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files.", )
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Pretrain Model selected in the list: [bert]")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path of pre-trained model")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.", )
    parser.add_argument("--wandb_desc", default="", type=str,
                        help="version description of wandb logging metrics.", )

    # Other parameters
    parser.add_argument('--markup', default='bio', type=str,
                        choices=['bios', 'bio', 'span'])
    parser.add_argument('--slot_decoder', default='softmax', type=str,
                        choices=['crf', 'softmax', 'global_pointer'])
    parser.add_argument('--slot_loss_type', default='ce', type=str,
                        choices=['multi_label_circle_loss', 'multi_label_focal_loss', 'label_smooth_ce', 'ce'])
    parser.add_argument('--intent_loss_type', default='ce', type=str,
                        choices=['multi_label_circle_loss', 'multi_class_focal_loss', 'multi_label_focal_loss', 'label_smooth_ce', 'ce', 'bce'])

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--eval_max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--do_export", action="store_true",
                        help="Whether to export lego model")                       
                        
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--freeze_pretrain", action="store_true",
                        help="Whether to freeze pretrain model.")
    # adversarial training
    parser.add_argument("--do_adv", action="store_true",
                        help="Whether to adversarial training.")

    parser.add_argument("--do_ema", action="store_true",
                        help="Whether to use ema.")
    parser.add_argument('--adv_epsilon', default=1.0, type=float,
                        help="Epsilon for adversarial.")
    parser.add_argument('--adv_name', default='word_embeddings', type=str,
                        help="name for adversarial layer.")

    parser.add_argument("--do_rdrop", action="store_true",
                        help="Whether to use r-drop.")
    parser.add_argument("--in_context", action="store_true",
                        help="Whether with context. In conversation, usully with context; In single query classify, usully no context.")
    parser.add_argument("--multi_label", action="store_true",
                        help="Whether multi label for classify task.")   
    
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--intent_loss_coef", default=1.0, type=float,
                        help="The loss coefficient of intent task.")
    parser.add_argument("--slot_loss_coef", default=1.0, type=float,
                        help="The loss coefficient of slot task.")
    parser.add_argument("--kl_loss_intent_coef", default=1.0, type=float,
                        help="The loss coefficient of intent kl task.")
    parser.add_argument("--kl_loss_slot_coef", default=1.0, type=float,
                        help="The loss coefficient of slot kl task.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--liner_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for linear layers.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam.")
    parser.add_argument("--dropout_rate", default=0., type=float,
                        help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Maximum gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps. Override num_train_epochs.", )

    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of warmup, E.g., 0.1 = 10% of training.")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log steps. Specially, -1 will logging every epoch")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X steps. Specially, -1 will saving every epoch")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with model_name ending and ending with step number", )
    parser.add_argument("--checkpoint", default="all", type=str,
                        help="Evaluate or Predict checkpoint")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using cuda when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite training and evaluation sets caches")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision instead of 32-bit", )
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See detail at https://nvidia.github.io/apex/amp.html", )
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    # context 截断方法
    parser.add_argument("--truncate_methods", default='hard_truncate', type=str, required=False,
                        help="the type of truncate concate_context selected in the list: "
                             "[punctuation,hard_truncate] , default : None", )
    # role embedding
    parser.add_argument("--speaker_segment", default=False, type=bool, required=False,
                        help="the type of embedding selected in the list: [True, False], default : False.", )
    parser.add_argument("--add_usr", action="store_true", help="add [USR] special token before query", )
    parser.add_argument("--num_cycles", type=int, default=2,
                        help="cosine restart scheduler num cycles")
    return parser
