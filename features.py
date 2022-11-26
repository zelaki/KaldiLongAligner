
import subprocess
import os
import re
from typing import Any, Dict
from utils import thirdparty_binary

MetaDict = Dict[str, Any]

# def thirdparty_binary(binary_name: str) -> str:
#     """
#     Generate full path to a given binary name
#     Notes
#     -----
#     With the move to conda, this function is deprecated as conda will manage the path much better
#     Parameters
#     ----------
#     binary_name: str
#         Executable to run
#     Returns
#     -------
#     str
#         Full path to the executable
#     """
#     bin_path = shutil.which(binary_name)
#     if bin_path is None:
#         if binary_name in ["fstcompile", "fstarcsort", "fstconvert"]:
#             raise Exception("Install Openfst and add it to path.")
#         elif binary_name in ["build-lm.sh"]:
#             raise Exception("Install IRSTLM and add it to path.")
#         else:
#             raise Exception(f"{binary_name} not in path")
#     if " " in bin_path:
#         return f'"{bin_path}"'
#     return bin_path




def make_safe(value: Any) -> str:
    """
    Transform an arbitrary value into a string
    Parameters
    ----------
    value: Any
        Value to make safe
    Returns
    -------
    str
        Safe value
    """
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


class Mfcc():
    """
    Multiprocessing function for generating MFCC features
    See Also
    --------
    :meth:`.AcousticCorpusMixin.mfcc`
        Main function that calls this function in parallel
    :meth:`.AcousticCorpusMixin.mfcc_arguments`
        Job method for generating arguments for this function
    :kaldi_src:`compute-mfcc-feats`
        Relevant Kaldi binary
    :kaldi_src:`extract-segments`
        Relevant Kaldi binary
    :kaldi_src:`copy-feats`
        Relevant Kaldi binary
    :kaldi_src:`feat-to-len`
        Relevant Kaldi binary
    Parameters
    ----------
    args: :class:`~montreal_forced_aligner.corpus.features.MfccArguments`
        Arguments for the function
    """
    progress_pattern = re.compile(r"^LOG.* Processed (?P<num_utterances>\d+) utterances")


    def __init__(self,
                args
                #  wav_path,
                # feats_scp_path,
                # mfcc_options,
                # pitch_options,
                # final_matrix,
                # log_path
        ):
        self.wav_path = args.wav_path
        self.feats_scp_path = args.feats_scp_path
        self.final_matrix = args.final_matrix
        self.mfcc_options = args.mfcc_options
        self.pitch_options = args.pitch_options
        self.log_path = args.log_path

    def make_feats(self, segment_path) -> None:
        """Run the function"""
        with open(self.log_path, "w") as log_file:
            # use_pitch = self.pitch_options.pop("use-pitch")
            use_pitch = False
            mfcc_base_command = [thirdparty_binary("compute-mfcc-feats"), "--verbose=2"]
            raw_ark_path = self.feats_scp_path.replace(".scp", ".ark")
            if os.path.exists(raw_ark_path):
                return
            for k, v in self.mfcc_options.items():
                mfcc_base_command.append(f"--{k.replace('_', '-')}={make_safe(v)}")
            if os.path.exists(segment_path):
                mfcc_base_command += ["ark:-", "ark:-"]
                seg_proc = subprocess.Popen(
                    [
                        thirdparty_binary("extract-segments"),
                        f"scp:{self.wav_path}",
                        segment_path,
                        "ark:-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    env=os.environ
                )
                comp1_proc = subprocess.Popen(
                    mfcc_base_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=seg_proc.stdout,
                    env=os.environ
                )
                cmvn_proc = subprocess.Popen(
                    [thirdparty_binary('apply-cmvn-sliding'), "--cmn-window=10000", "--center=true", "ark:-", "ark:-"],
                    stdin=comp1_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=os.environ
                )

                splice_proc = subprocess.Popen(
                    [thirdparty_binary('splice-feats'), "--left-context=5", "--right-context=5", "ark:-", "ark:-"],
                    stdin=cmvn_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=os.environ
                )

                comp_proc = subprocess.Popen(
                    [thirdparty_binary('transform-feats'), self.final_matrix, "ark:-", "ark:-"],
                    stdin=splice_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=os.environ
                )


            else:
                mfcc_base_command += [f"scp,p:{self.wav_path}", "ark:-"]
                comp_proc = subprocess.Popen(
                    mfcc_base_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=os.environ,
                )
            if use_pitch:
                pitch_base_command = [
                    thirdparty_binary("compute-and-process-kaldi-pitch-feats"),
                    "--verbose=2",
                ]
                for k, v in self.pitch_options.items():
                    pitch_base_command.append(f"--{k.replace('_', '-')}={make_safe(v)}")
                    if k == "delta-pitch":
                        pitch_base_command.append(f"--delta-pitch-noise-stddev={make_safe(v)}")
                pitch_command = " ".join(pitch_base_command)
                if os.path.exists(segment_path):
                    segment_command = (
                        f'extract-segments scp:"{self.wav_path}" "{segment_path}" ark:- | '
                    )
                    pitch_input = "ark:-"
                else:
                    segment_command = ""
                    pitch_input = f'scp:"{self.wav_path}"'
                pitch_feat_string = (
                    f"ark,s,cs:{segment_command}{pitch_command} {pitch_input} ark:- |"
                )
                length_tolerance = 2
                paste_proc = subprocess.Popen(
                    [
                        thirdparty_binary("paste-feats"),
                        f"--length-tolerance={length_tolerance}",
                        "ark:-",
                        pitch_feat_string,
                        "ark:-",
                    ],
                    stdin=comp_proc.stdout,
                    env=os.environ,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                )
                copy_proc = subprocess.Popen(
                    [
                        thirdparty_binary("copy-feats"),
                        "--verbose=2",
                        "--compress=true",
                        "ark:-",
                        f"ark,scp:{raw_ark_path},{self.feats_scp_path}",
                    ],
                    stdin=paste_proc.stdout,
                    stderr=subprocess.PIPE,
                    env=os.environ,
                    encoding="utf8",
                )
            else:
                copy_proc = subprocess.Popen(
                    [
                        thirdparty_binary("copy-feats"),
                        "--verbose=2",
                        "--compress=true",
                        "ark:-",
                        f"ark,scp:{raw_ark_path},{self.feats_scp_path}",
                    ],
                    stdin=comp_proc.stdout,
                    stderr=log_file,
                    env=os.environ,
                    encoding="utf8",
                )
            for line in comp_proc.stderr:
                line = line.strip().decode("utf8")
                log_file.write(line + "\n")
                m = self.progress_pattern.match(line)
                if m:
                    cur = int(m.group("num_utterances"))
                    # increment = cur - processed
                    # processed = cur
                    # yield increment
            # self.check_call(copy_proc)


# if __name__ == '__main__':
    # features_config = FeatureConfigMixin()
    # feat_ops = features_config.mfcc_options
    # pitch_ops = features_config.pitch_options

    # # print(ops.items())

    # mfcc=Mfcc(wav_path='wav.scp', segment_path='segments', feats_scp_path='feats.scp', mfcc_options=feat_ops, pitch_options=pitch_ops, final_matrix='final.mat',log_path='log.txt')
    # mfcc.make_feats()