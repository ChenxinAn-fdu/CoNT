import argparse
import os


def run(cmd):
    print(cmd)
    os.system(cmd)


def preprocess_cmd():
    TEXT = "iwslt14.tokenized.de-en"
    cmd = f"python preprocess.py --source-lang {args.source} --target-lang {args.target} \
    --trainpref {TEXT}/train --validpref {TEXT}/valid --testpref {TEXT}/test \
    --destdir data-bin/iwslt14.tokenized.{args.source}-{args.target} \
    --workers 20"
    run(cmd)


def gen_cmd():
    if args.avg_ckpt:
        assert os.path.isdir(args.save_path)
        avg_ckpt = os.path.join(args.save_path, "average.pt")
        run(f"python fairseq/scripts/average_checkpoints.py --inputs {args.save_path} --num-epoch-checkpoints 3 --output {avg_ckpt}")
        args.save_path = avg_ckpt
    generate_cmd = f"CUDA_VISIBLE_DEVICES={args.gpus} python cont_generate.py \
        data-bin/iwslt14.tokenized.{args.source}-{args.target} \
        --path {args.save_path} --task translation_cl  --batch-size 256 --remove-bpe "
    override_param = " --model-overrides '{\"lenpen\":1.0, \"beam_size\":12 }'"
    result_path = args.save_path.replace(".pt", ".out")
    pipe_param = f" | tee {result_path}"
    run(generate_cmd + override_param + pipe_param)


def score_cmd():
    out_file = args.out_file
    ref_file = out_file.replace(".out", ".ref")
    sys_file = out_file.replace(".out", ".sys")
    run(
        f"grep ^T {out_file}" + " | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' " + f"> {ref_file}")
    run(
        f"grep ^H {out_file}" + " | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' " + f"> {sys_file}")

    run(f"python score.py --sys {sys_file} --ref {ref_file}")


def train_cmd():
    shared_cmd = f"CUDA_VISIBLE_DEVICES={args.gpus}  python train.py \
      data-bin/iwslt14.tokenized.{args.source}-{args.target}  --task translation_cl  \
      --arch transformer_cont_iwslt --share-decoder-input-output-embed \
      --optimizer adam  --clip-norm 0.0 --warmup-updates 4000 \
      --lr 5e-4  --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt \
      --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
      --dropout 0.3 --weight-decay 0.0001  --fp16  \
      --save-dir /nvme/anchenxin/repro/iwslt14-{args.source}-{args.target}-new/ \
      --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu \
      --keep-interval-updates 2  --log-interval 10 --keep-last-epochs 1 \
      --maximize-best-checkpoint-metric --best-checkpoint-metric bleu \
      --diverse_bias 2.5 --max_len_a 1.2 --max_len_b 10 --beam_size 12  "
    warmup_cmd = f" --max-epoch  {int(args.warmup_epochs)}  --keep-best-checkpoints 1 --find-unused-parameters \
      --max-tokens {args.warmup_max_tokens}  --update-freq {args.warmup_accum} --warmup 1 "
    if args.warmup:
        run(shared_cmd + warmup_cmd)
    cont_cmd = f" --validate-interval-updates 1000  --alpha {args.alpha}  --save-interval-updates 40 --max_sample_num 32 --from_hypo 0.75  \
      --max-epoch {args.total_epochs}  --keep_dropout 0  --keep-best-checkpoints 3 --warmup 0 \
      --update-freq {args.accum} --max-tokens {args.max_tokens}  --reset-dataloader"
    run(shared_cmd + cont_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='running script python wrapper'
    )
    # config
    parser.add_argument('--mode', choices=["preprocess", "train", "gen", "score"])
    parser.add_argument('--source', default="de")
    parser.add_argument('--target', default="en")
    parser.add_argument('--gpus', default="0,1,2,3")

    # training parameter
    parser.add_argument('--warmup', action="store_true")
    parser.add_argument('--warmup_max_tokens', default=1024, type=int)
    parser.add_argument('--max_tokens', default=3200, type=int)
    parser.add_argument('--warmup_accum', default=1, type=int)
    parser.add_argument('--accum', default=4, type=int)
    parser.add_argument('--warmup_epochs', default=50, type=int)
    parser.add_argument('--total_epochs', default=25, type=int)

    parser.add_argument('--alpha', default=0.5, type=float)

    # inference
    parser.add_argument('--out_file', default=None)
    parser.add_argument('--avg_ckpt', action="store_true")
    parser.add_argument('--save_path', default="")
    args = parser.parse_args()
    eval(f"{args.mode}_cmd()")
