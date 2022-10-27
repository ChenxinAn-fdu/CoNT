import argparse
import os


def run(cmd):
    print(cmd)
    os.system(cmd)


def preprocess_cmd():
    TEXT = f"wmt16.tokenized.{args.source}-{args.target}"
    cmd = f"python preprocess.py --source-lang {args.source} --target-lang {args.target} \
    --trainpref {TEXT}/train --validpref {TEXT}/valid --testpref {TEXT}/test \
    --destdir data-bin/wmt14.tokenized.{args.source}-{args.target} \
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
    override_param = " --model-overrides '{\"beam_size\":8 }'"
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
      data-bin/wmt14.tokenized.{args.source}-{args.target}  --task translation_cl  \
      --arch transformer_cont_wmt --share-decoder-input-output-embed \
      --optimizer adam  --clip-norm 0.0 --warmup-updates 4000 \
      --lr 5e-4  --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --stop-min-lr '1e-09' \
      --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
      --dropout 0.3 --weight-decay 0.0001  --warmup-init-lr '1e-07' --fp16  \
      --save-dir /nvme/anchenxin/repro/wmt-{args.source}-{args.target}-new/ \
      --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu \
      --keep-interval-updates 2  --log-interval 10 --keep-last-epochs 1 \
      --maximize-best-checkpoint-metric --best-checkpoint-metric bleu \
      --diverse_bias 2.5 --max_len_a 1.0 --max_len_b 50 --beam_size 8  "
    warmup_cmd = f" --max-epoch  {int(args.warmup_epochs)}  --keep-best-checkpoints 1 --find-unused-parameters \
      --max-tokens {args.warmup_max_tokens}  --update-freq {args.warmup_accum} --warmup 1 "
    if args.warmup:
        run(shared_cmd + warmup_cmd)
    # set max_sample_num to 32 will provide better results
    cont_cmd = f" --validate-interval-updates 1000  --alpha {args.alpha}  --save-interval-updates 80 --max_sample_num 16 --from_hypo 0.75  \
      --max-epoch {args.total_epochs}  --keep_dropout 1  --keep-best-checkpoints 3 --warmup 0 \
      --update-freq {args.accum} --max-tokens {args.max_tokens}  --reset-dataloader"
    run(shared_cmd + cont_cmd)


## wmt requires a large batch size
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run cmd python wrapper'
    )
    # config
    parser.add_argument('--mode', choices=["preprocess", "train", "gen", "score"])
    parser.add_argument('--source', default="en")
    parser.add_argument('--target', default="de")
    parser.add_argument('--gpus', default="0,1,2,3")

    # training parameter
    parser.add_argument('--warmup', action="store_true")
    parser.add_argument('--warmup_max_tokens', default=16000, type=int)
    parser.add_argument('--max_tokens', default=3200, type=int)
    parser.add_argument('--warmup_accum', default=1, type=int)
    parser.add_argument('--accum', default=8, type=int)
    parser.add_argument('--warmup_epochs', default=150, type=int)
    parser.add_argument('--total_epochs', default=10, type=int)

    parser.add_argument('--alpha', default=0.3, type=float)

    # inference
    parser.add_argument('--out_file', default=None)
    parser.add_argument('--avg_ckpt', action="store_true")
    parser.add_argument('--save_path', default="")
    args = parser.parse_args()
    eval(f"{args.mode}_cmd()")
