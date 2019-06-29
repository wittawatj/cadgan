"""
An executable script to generate commands (or scripts) for submitting to a
computing cluster.
"""
import cmdprod as cp

# import numpy as np


def args_spec(
    g_path, g_type, logdir, list_cond_path, list_extractor_type=["vgg_face"], list_extractor_layers=["8 17 26 35"]
):
    """
    Return a specification arguments (including candidate values) for 
    an experiment with Lars's models.
    
    #!/bin/bash

    python3 run_lars_gkmm.py \
        --extractor_type vgg_face \
        --extractor_layers 8 17 26 35 \
        --texture 1\
        --depth_process no \
        --g_path gan_data/celebAHQ_00/chkpts/model.pt \
        --g_type celebAHQ.yaml \
        --g_min -1.0 \
        --g_max 1.0 \
        --logdir log_lars_celeba_vggface/ \
        --device gpu \
        --n_sample 1 \
        --n_opt_iter 3000 \
        --lr 1e-1 \
        --seed 9 \
        --img_log_steps 10 \
        --cond_path  c.txt\
        --kernel imq \
        --kparams -0.5 1e+2 \
    """
    # constants
    p_extractor_type = cp.Param("extractor_type", list_extractor_type)
    p_extractor_layers = cp.Param("extractor_layers", list_extractor_layers)
    p_texture = cp.Param("texture", [0])
    p_depth_process = cp.Param("depth_process", ["no"])
    p_g_path = cp.Param("g_path", [g_path])
    p_g_type = cp.Param("g_type", [g_type])
    p_g_min = cp.Param("g_min", [-1.0])
    p_g_max = cp.Param("g_max", [1.0])
    p_logdir = cp.Param("logdir", [logdir])
    p_device = cp.Param("device", ["gpu"])
    p_n_opt_iter = cp.Param("n_opt_iter", [3000])
    p_img_log_steps = cp.Param("img_log_steps", [10])

    # with some candidate values to try
    p_n_sample = cp.Param("n_sample", [1, 2])
    p_lr = cp.Param("lr", [1e-2, 1e-1])
    p_seed = cp.Param("seed", [8, 9])
    p_cond_path = cp.Param("cond_path", list_cond_path)

    p_kgroup = cp.ParamGroup(
        ["kernel", "kparams"],
        [
            ("linear", []),
            # ('gauss', 10.0),
            ("imq", [-0.5, 1.0]),
            ("imq", [-0.5, 100.0]),
        ],
    )
    args = cp.Args(
        [
            p_extractor_type,
            p_extractor_layers,
            p_texture,
            p_depth_process,
            p_g_path,
            p_g_type,
            p_g_min,
            p_g_max,
            p_logdir,
            p_device,
            p_n_opt_iter,
            p_img_log_steps,
            #
            p_n_sample,
            p_lr,
            p_seed,
            p_cond_path,
            p_kgroup,
        ]
    )
    return args


def main():
    bundle1 = dict(
        g_path="gan_data/celebAHQ_00/chkpts/model.pt",
        g_type="celebAHQ.yaml",
        logdir="log_lars_celeba_batch/",
        list_cond_path=["c.txt"],
        list_extractor_type=["vgg_face"],
        list_extractor_layers=["17", "8 17 26 35"],
    )
    args = args_spec(**bundle1)

    line_begin = "python3 ../run_lars_gkmm.py "
    # potentially the destination script folder name should depend on the type
    # of bundle
    dest_path = "cmd_celeba1"
    token_path = dest_path
    # print to stdout
    # args_processor = cp.APPrint(prefix=line_begin, suffix='\n')
    # args_processor.iaf.value_formatter.list_value_sep = ', '
    # args_processor(args)

    # generate Bash files in a folder
    iaf = cp.IAFArgparse(
        # separtor for different pair-value pairs
        pv_sep=" \\\n"
    )
    args_file_proc = cp.APPerBashFile(
        dest_path=dest_path,
        token_path=token_path,
        iaf=iaf,
        file_begin="#!/bin/bash",
        file_end="# end of script",
        line_begin=line_begin,
        line_end="",
    )
    # for values which are lists, separate items in the list with the following
    # string
    args_file_proc.iaf.value_formatter.list_value_sep = " "
    args_file_proc(args)


if __name__ == "__main__":
    main()
