import argparse


def get_ti_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )

    parser.add_argument(
        "--datadir_in_name",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Prepend the final directory in the data_root to the output directory name")

    parser.add_argument(
        "--use_facial_loss",
        action='store_true',
        help="Use facial loss when fine tuning text-to-image model"
    )

    parser.add_argument(
        "--use_random_prompt",
        action='store_true',
        help="Use random prompt when fine tuning text-to-image model"
    )

    parser.add_argument(
        "--descriptive_p",
        type=float,
        default=0.2,
        help="concatenate descriptive words with given probabilities"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="ratio of what momentum vector update embedding vector"
    )

    parser.add_argument("--actual_resume",
        type=str,
        required=True,
        help="Path to model to actually resume from")

    parser.add_argument("--data_root",
        type=str,
        required=True,
        help="Path to directory with training images")

    parser.add_argument("--embedding_manager_ckpt",
        type=str,
        default="",
        help="Initialize embedding manager from a checkpoint")

    parser.add_argument("--placeholder_string",
        type=str,
        help="Placeholder string which will be used to denote the concept in future prompts. Overwrites the config options.")

    parser.add_argument("--init_word",
        type=str,
        help="Word to use as source for initial token embedding")

    parser.add_argument(
        "--negative_prompt",
        type=str,
        nargs="?",
        default="deformed, cripple, ugly, additional arms, additional legs, additional head, two heads, multiple people, group of people black and white, grayscale, collage, cropped head, out of frame, blurry, group of people, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, bad anatomy, bad proportions, extra limbs, disfigured, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers",
        help="the negative prompt to sample during training"
    )

    parser.add_argument(
        "--class_word",
        type=str,
        help="Word to embed next to the identifier"
    )

    return parser


def get_txt2img_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "--name",
        type=str,
        nargs="?",
        default="woman",
        help="the sampled image name"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        nargs="?",
        default="deformed, cripple, ugly, additional arms, additional legs, additional head, two heads, multiple people, group of people black and white, grayscale, collage, cropped head, out of frame, blurry, group of people, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, bad anatomy, bad proportions, extra limbs, disfigured, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers",
        help="the negative prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--embedding_path",
        type=str,
        help="Path to a pre-trained embedding manager checkpoint")

    opt = parser.parse_args()

    return opt
