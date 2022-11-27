import subprocess
import os
import math
import io
from typing import TextIO
from aligner.utils import thirdparty_binary

def compose_lg(dictionary_path: str, small_g_path: str, lg_path: str, log_file: TextIO) -> None:
    """
    Compose an LG.fst
    See Also
    --------
    :kaldi_src:`fsttablecompose`
        Relevant Kaldi binary
    :kaldi_src:`fstdeterminizestar`
        Relevant Kaldi binary
    :kaldi_src:`fstminimizeencoded`
        Relevant Kaldi binary
    :kaldi_src:`fstpushspecial`
        Relevant Kaldi binary
    Parameters
    ----------
    dictionary_path: str
        Path to a lexicon fst file
    small_g_path: str
        Path to the small language model's G.fst
    lg_path: str
        Output path to LG.fst
    log_file: TextIO
        Log file handler to output logging info to
    """
    if os.path.exists(lg_path):
        return
    compose_proc = subprocess.Popen(
        [thirdparty_binary("fsttablecompose"), dictionary_path, small_g_path],
        stderr=log_file,
        stdout=subprocess.PIPE,
        env=os.environ
    )

    determinize_proc = subprocess.Popen(
        [
            thirdparty_binary("fstdeterminizestar"),
            "--use-log=true",
        ],
        stdin=compose_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=log_file,
        env=os.environ
    )

    minimize_proc = subprocess.Popen(
        [thirdparty_binary("fstminimizeencoded")],
        stdin=determinize_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=log_file,
        env=os.environ
    )

    push_proc = subprocess.Popen(
        [thirdparty_binary("fstpushspecial"), "-", lg_path],
        stdin=minimize_proc.stdout,
        stderr=log_file,
        env=os.environ
    )
    push_proc.communicate()


def get_tree_info(tree_path, log_file, info):
    tree_info_proc = subprocess.Popen(
        [thirdparty_binary("tree-info"), tree_path],
        stderr=log_file,
        stdout=subprocess.PIPE,
        env=os.environ
    )
    grep_proc = subprocess.Popen(
        [thirdparty_binary("grep"), info],
        stdin=tree_info_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=log_file,
        env=os.environ

    )
    info = grep_proc.stdout.read().decode("utf-8").rstrip().split()[1]
    return info

def compose_clg(
    in_disambig: str,
    out_disambig: str,
    context_width: int,
    central_pos: int,
    ilabels_temp: str,
    lg_path: str,
    clg_path: str,
    log_file: TextIO,
) -> None:
    """
    Compose a CLG.fst
    See Also
    --------
    :kaldi_src:`fstcomposecontext`
        Relevant Kaldi binary
    :openfst_src:`fstarcsort`
        Relevant OpenFst binary
    Parameters
    ----------
    in_disambig: str
        Path to read disambiguation symbols file
    out_disambig: str
        Path to write disambiguation symbols file
    context_width: int
        Context width of the acoustic model
    central_pos: int
        Central position of the acoustic model
    ilabels_temp:
        Temporary file for ilabels
    lg_path: str
        Path to a LG.fst file
    clg_path:
        Path to save CLG.fst file
    log_file: TextIO
        Log file handler to output logging info to
    """
    compose_proc = subprocess.Popen(
        [
            thirdparty_binary("fstcomposecontext"),
            f"--context-size={context_width}",
            f"--central-position={central_pos}",
            f"--read-disambig-syms={in_disambig}",
            f"--write-disambig-syms={out_disambig}",
            ilabels_temp,
            lg_path,
        ],
        stdout=subprocess.PIPE,
        stderr=log_file,
    )
    sort_proc = subprocess.Popen(
        [thirdparty_binary("fstarcsort"), "--sort_type=ilabel", "-", clg_path],
        stdin=compose_proc.stdout,
        stderr=log_file,
        env=os.environ,
    )
    sort_proc.communicate()



def compose_hclg(
    model_directory: str,
    ilabels_temp: str,
    transition_scale: float,
    clg_path: str,
    hclga_path: str,
    log_file: TextIO,
) -> None:
    """
    Compost HCLG.fst for a dictionary
    See Also
    --------
    :kaldi_src:`make-h-transducer`
        Relevant Kaldi binary
    :kaldi_src:`fsttablecompose`
        Relevant Kaldi binary
    :kaldi_src:`fstdeterminizestar`
        Relevant Kaldi binary
    :kaldi_src:`fstrmsymbols`
        Relevant Kaldi binary
    :kaldi_src:`fstrmepslocal`
        Relevant Kaldi binary
    :kaldi_src:`fstminimizeencoded`
        Relevant Kaldi binary
    :openfst_src:`fstarcsort`
        Relevant OpenFst binary
    Parameters
    ----------
    model_directory: str
        Model working directory with acoustic model information
    ilabels_temp: str
        Path to temporary ilabels file
    transition_scale: float
        Transition scale for the fst
    clg_path: str
        Path to CLG.fst file
    hclga_path: str
        Path to save HCLGa.fst file
    log_file: TextIO
        Log file handler to output logging info to
    """
    model_path = os.path.join(model_directory, "final.mdl")
    tree_path = os.path.join(model_directory, "tree")
    ha_path = hclga_path.replace("HCLGa", "Ha")
    ha_out_disambig = hclga_path.replace("HCLGa", "disambig_tid")
    make_h_proc = subprocess.Popen(
        [
            thirdparty_binary("make-h-transducer"),
            f"--disambig-syms-out={ha_out_disambig}",
            f"--transition-scale={transition_scale}",
            ilabels_temp,
            tree_path,
            model_path,
            ha_path,
        ],
        stderr=log_file,
        stdout=log_file,
        env=os.environ,
    )
    make_h_proc.communicate()

    compose_proc = subprocess.Popen(
        [thirdparty_binary("fsttablecompose"), ha_path, clg_path],
        stderr=log_file,
        stdout=subprocess.PIPE,
        env=os.environ,
    )

    determinize_proc = subprocess.Popen(
        [thirdparty_binary("fstdeterminizestar"), "--use-log=true"],
        stdin=compose_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=log_file,
        env=os.environ,
    )
    rmsymbols_proc = subprocess.Popen(
        [thirdparty_binary("fstrmsymbols"), ha_out_disambig],
        stdin=determinize_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=log_file,
        env=os.environ,
    )
    rmeps_proc = subprocess.Popen(
        [thirdparty_binary("fstrmepslocal")],
        stdin=rmsymbols_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=log_file,
        env=os.environ,
    )
    minimize_proc = subprocess.Popen(
        [thirdparty_binary("fstminimizeencoded"), "-", hclga_path],
        stdin=rmeps_proc.stdout,
        stderr=log_file,
        env=os.environ,
    )
    minimize_proc.communicate()


def generate_text_transducer(lm_text: str, words_path: str, g_text_path:str, skip: bool=False):
    """
    Compost G.txt. i.e. text format for linear transduser
    Parameters
    ----------
    lm_text: str
        Path to file containing one line of text
    words_path: str
        Path to words.txt 
    transition_scale: float
        Transition scale for the fst
    g_text_path: str
        Path to output G.txt
    skip: bool
        If True creates liner transducer that allows insertions and deletions
    """
    sym2int = {}
    with open(words_path, 'r') as fd:
        lines = [ln.strip().split() for ln in fd]
    for word, id in lines:
        sym2int[word] = id
    with open(lm_text,'r') as f:
        input_contents=f.readline()

    words=input_contents.split(' ')
    idx=1
    with open(g_text_path, 'w', encoding='utf-8') as f:
        for w in words:
            w=w.strip()
            if w not in sym2int:
                w = '<UNK>'
            if(skip != True):
                f.write('{} {} {} {} {}\n'.format(idx-1, idx, w, w, '0'))
            else:
                f.write('{} {} {} {} {}\n'.format(idx-1, idx-1, w, w, -math.log(0.05,10)))
                f.write('{} {} {} {} {}\n'.format(idx-1, idx, w, w, -math.log(0.9,10)))
                f.write('{} {} {} {} {}\n'.format(idx-1, idx, '<eps>', '<eps>', -math.log(0.05,10)))

            idx+=1

# if __name__ == '__main__':
#     words_path = '/home/theokouz/kaldi/egs/betterReading/s5/data/lang_kids_new/words.txt'

#     generate_text_transducer('lm_text', words_path,'G.txt')



