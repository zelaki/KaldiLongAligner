import subprocess
from aligner.utils import thirdparty_binary, ctmEntry
import os 
from typing import Dict, List



class Transcriber():

    def __init__(
        self,
        args
        ) -> None:
        self.word_insertion_penalty = args.word_insertion_penalty
        self.frame_shift = args.frame_shift
        self.acoustic_scale = args.acoustic_scale
        self.max_active = args.max_active
        self.beam = args.beam
        self.lattice_beam = args.lattice_beam
        self.words_path = args.words_path
        self.final_model = args.final_model
        self.final_mat = args.final_mat
        self.hclg_path = args.hclg_path
        self.spk2utt = args.spk2utt
        self.feats = args.feats
        self.cmvn_ark = args.cmvn_ark
        self.cmvn_scp = args.cmvn_scp
        self.left_context = args.left_context
        self.right_context = args.right_context
        self.word_boundary_int = args.word_boundary_int
        self.feature_string = args.feature_string
        self.lat_path = args.lat_path
        self.fmllr_update_type = args.fmllr_update_type
        self.sil_phones = args.sil_phones
        self.silence_weight = args.silence_weight
        self.pre_trans_path = args.pre_trans_path
        self.temp_trans_path = args.temp_trans_path
        self.tmp_lat = args.tmp_lat
        self.trans_path = args.trans_path
        self.final_lat_path = args.final_lat_path

        self.int2sym_dictionary = {}
        with open(self.words_path, 'r') as f:
            lines = [ln.rstrip().split() for ln in f.readlines()]  
        for word, integer in lines:
            self.int2sym_dictionary[integer] = word


    def compute_cmvn_stats(
        self,
    ) -> None:
        subprocess.call(
            [
                thirdparty_binary("compute-cmvn-stats"),
                f"--spk2utt=ark:{self.spk2utt}",
                f"scp:{self.feats}",
                f"ark,scp:{self.cmvn_ark},{self.cmvn_scp}",
            ],
            env=os.environ,
        )




    def generate_lattices(
        self,
        lat_path,
        feature_string,
        determinize="true"
        ) -> None:
        decode_proc = subprocess.Popen(
            [
                thirdparty_binary("gmm-latgen-faster"),
                f"--max-active={self.max_active}",
                f"--beam={self.beam}",
                f"--lattice-beam={self.lattice_beam}",
                "--allow-partial=true",
                f"--determinize-lattice={determinize}",
                f"--word-symbol-table={self.words_path}",
                f"--acoustic-scale={self.acoustic_scale}",
                self.final_model,
                self.hclg_path,
                feature_string,
                f"ark:{lat_path}",
            ],
            # stderr=subprocess.PIPE,
            env=os.environ,
            encoding="utf8",
        )
        decode_proc.communicate()




    def lattice_to_ctm(
        self,
        lat_path
    ):

        scale_proc =  subprocess.Popen(
            [
                thirdparty_binary("lattice-add-penalty"),
                f"--word-ins-penalty={self.word_insertion_penalty}",
                f'ark:{lat_path}',
                "ark:-"

            ],
            stdout=subprocess.PIPE,
            env=os.environ,
            encoding="utf8"
        )
        
        one_best_proc = subprocess.Popen(
            [
                thirdparty_binary("lattice-1best"),
                f"--acoustic-scale={self.acoustic_scale}",
                "ark:-",
                "ark:-"
            ],
            stdin=scale_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
            encoding="utf8"
        )

        align_words_proc = subprocess.Popen(
            [
                thirdparty_binary("lattice-align-words"),
                self.word_boundary_int,
                self.final_model,
                "ark:-",
                "ark:-"
            ],
            stdin=one_best_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
            encoding="utf8"
        )
        
        nbest_to_ctm_proc = subprocess.Popen(
            [
                thirdparty_binary("nbest-to-ctm"),
                f"--frame-shift={self.frame_shift}",
                "ark:-",
                "-"
            ],
            stdin=align_words_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
            encoding="utf8"
        )
        out, err = nbest_to_ctm_proc.communicate()
        out = [ln.split() for ln in out.split('\n') if ln != '']

        return out


    def decode(
        self,
        working_dir
    ) -> None:
        
        feature_string = self.feature_string.format(
            working_dir=working_dir,
            final_mat = self.final_mat,
            left_context = self.left_context,
            right_context = self.right_context
            )

        self.compute_cmvn_stats()
        self.generate_lattices(self.lat_path, feature_string)
        self.calculate_initial_fmllr(feature_string)
        feats_first_pass = feature_string+f" transform-feats --utt2spk=ark:{self.spk2utt} ark:{self.pre_trans_path} ark:- ark:- |"
        self.generate_lattices(self.tmp_lat, feats_first_pass, determinize="false")
        self.calculate_final_fmllr(feats_first_pass)
        feats_final_pass = feature_string+f" transform-feats --utt2spk=ark:{self.spk2utt} ark:{self.trans_path} ark:- ark:- |"
        self.rescore_fmllr(feats_final_pass)
        out = self.lattice_to_ctm(self.final_lat_path)
        words_int = [ln[4] for ln in out]
        words_sym = [self.int2sym_dictionary[word] for word in words_int]
        words_ctm_sym = [
            ctmEntry(
                word=self.int2sym_dictionary[ln[4]],
                onset=float(ln[2]),
                duration=float(ln[3])
            ) 
            for ln in out
            ]
        return words_sym, words_ctm_sym

    def calculate_final_fmllr(
        self,
        feature_string
    ) -> None:
        determinize_proc = subprocess.Popen(
            [
                thirdparty_binary("lattice-determinize-pruned"),
                f"--acoustic-scale={self.acoustic_scale}",
                "--beam=4.0",
                f"ark:{self.tmp_lat}",
                "ark:-",
            ],
            stdout=subprocess.PIPE,
            env=os.environ,
        )

        latt_post_proc = subprocess.Popen(
            [
                thirdparty_binary("lattice-to-post"),
                f"--acoustic-scale={self.acoustic_scale}",
                "ark:-",
                "ark:-",
            ],
            stdin=determinize_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        weight_silence_proc = subprocess.Popen(
            [
                thirdparty_binary("weight-silence-post"),
                f"{self.silence_weight}",
                self.sil_phones,
                self.final_model,
                "ark:-",
                "ark:-",
            ],
            stdin=latt_post_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        fmllr_proc = subprocess.Popen(
            [
                thirdparty_binary("gmm-est-fmllr"),
                f"--fmllr-update-type={self.fmllr_update_type}",
                f"--spk2utt=ark:{self.spk2utt}",
                self.final_model,
                feature_string,
                "ark,s,cs:-",
                f"ark:{self.temp_trans_path}",
            ],
            stdin=weight_silence_proc.stdout,
            stderr=subprocess.PIPE,
            env=os.environ,
            encoding="utf8",
        )
        fmllr_proc.communicate()

        compose_transforms_proc = subprocess.Popen(
            [
                thirdparty_binary("compose-transforms"),
                "--b-is-affine=true",
                f"ark:{self.temp_trans_path}",
                f"ark:{self.pre_trans_path}",
                f"ark:{self.trans_path}"
            ]
        )
        compose_transforms_proc.communicate()


    def calculate_initial_fmllr(
        self,
        feature_string
    ) -> None:

        latt_post_proc = subprocess.Popen(
            [
                thirdparty_binary("lattice-to-post"),
                f"--acoustic-scale={self.acoustic_scale}",
                f"ark:{self.lat_path}",
                "ark:-",
            ],
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        weight_silence_proc = subprocess.Popen(
            [
                thirdparty_binary("weight-silence-post"),
                f"{self.silence_weight}",
                self.sil_phones,
                self.final_model,
                "ark,s,cs:-",
                "ark:-",
            ],
            stdin=latt_post_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        gmm_gpost_proc = subprocess.Popen(
            [
                thirdparty_binary("gmm-post-to-gpost"),
                self.final_model,
                feature_string,
                "ark,s,cs:-",
                "ark:-",
            ],
            stdin=weight_silence_proc.stdout,
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        fmllr_proc = subprocess.Popen(
            [
                thirdparty_binary("gmm-est-fmllr-gpost"),
                f"--fmllr-update-type={self.fmllr_update_type}",
                f"--spk2utt=ark:{self.spk2utt}",
                self.final_model,
                feature_string,
                "ark,s,cs:-",
                f"ark:{self.pre_trans_path}",
            ],
            stdin=gmm_gpost_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
            encoding="utf8",
        )
        fmllr_proc.communicate()

    def rescore_fmllr(
        self,
        feature_string
    ) -> None:
        rescore_proc = subprocess.Popen(
            [
                thirdparty_binary("gmm-rescore-lattice"),
                self.final_model,
                f"ark:{self.tmp_lat}",
                feature_string,
                "ark:-",
            ],
            stdout=subprocess.PIPE,
            env=os.environ,
        )
        determinize_proc = subprocess.Popen(
            [
                thirdparty_binary("lattice-determinize-pruned"),
                f"--acoustic-scale={self.acoustic_scale}",
                f"--beam={self.lattice_beam}",
                "ark:-",
                f"ark:{self.final_lat_path}",
            ],
            stdin=rescore_proc.stdout,
            stderr=subprocess.PIPE,
            encoding="utf8",
            env=os.environ,
        )
        determinize_proc.communicate
        determinize_proc.wait()





if __name__ == '__main__':

    from aligner.config import create_trascriber_args_n
    from aligner.utils import read_yaml_file
    config = read_yaml_file('settings.yaml')
    args = create_trascriber_args_n('working_dir', 'model', config)
    transcriber = Transcriber(args)
    print(transcriber.decode(working_dir='working_dir'))

