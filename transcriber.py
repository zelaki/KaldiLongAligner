import typing
import os
import subprocess
from typing import List
from graph import compose_clg, compose_hclg, compose_lg, generate_text_transducer, get_tree_info
from kaldi.asr import GmmLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.util.table import SequentialMatrixReader
from kaldi.lat.align import WordBoundaryInfoNewOpts, WordBoundaryInfo
from kaldi.alignment import GmmAligner
from utils import thirdparty_binary, ctmEntry, HCLGArgs, TranscriberArgs
from features import Mfcc


final_model = '/home/theokouz/kaldi/egs/betterReading/s5/exp/tri3b_bs/final.mdl'
tree = '/home/theokouz/kaldi/egs/betterReading/s5/exp/tri3b_bs/tree'
L = '/home/theokouz/kaldi/egs/betterReading/s5/data/lang_kids_new/L.fst'
words = '/home/theokouz/kaldi/egs/betterReading/s5/data/lang_kids_new/words.txt'
disambig = '/home/theokouz/kaldi/egs/betterReading/s5/data/lang_kids_new/phones/disambig.int'
phones = '/home/theokouz/kaldi/egs/betterReading/s5/data/lang_kids/phones.txt'
word_bound = '/home/theokouz/kaldi/egs/betterReading/s5/data/lang_kids_new/phones/word_boundary.int'
hclg = '/home/theokouz/kaldi/egs/betterReading/s5/exp/tri3b_bs/graph/HCLG.fst'







class CreateHCLG():
    """
    Class used to cleate HCLG decoding graph.

    Parameters
    ----------

    disambig_L_path: str
        Path to disambig_L
    disambig_int_path: str
        Path to disambig.int
    words_path: str
        Path to words.txt
    wording_directory: str
        Path to working directory
    final_model_path: str
        Path to final.mdl
    """
    
    def __init__(
    self,
    args: HCLGArgs
    # disambig_L_path: str,
    # disambig_int_path: str,
    # words_path: str,
    # working_directory: str,
    # final_model_path: str,
):
        self.disambig_L_path = args.disambig_L_path
        self.disambig_int_path = args.disambig_int_path
        self.working_directory = args.working_directory
        self.log_file_path = os.path.join(args.working_directory, 'log.txt')
        self.final_model_path = args.final_model_path
        self.graph_directory = os.path.join(args.working_directory, 'graph_dir')
        self.lm_directory = os.path.join(args.working_directory, 'lm_dir')
        self.transition_scale = '1.0'
        self.self_loop_scale = '0.1'
        self.words_path = args.words_path
        self.g_text_path = os.path.join(self.graph_directory, 'G.txt')
        self.g_path = os.path.join(self.graph_directory, 'G.fst')
        self.lg_path = os.path.join(self.graph_directory, 'LG.fst')
        self.clg_path = os.path.join(self.graph_directory, 'CLG.fst')
        self.hclga_path = os.path.join(self.graph_directory, 'HCLGa.fst')
        self.hclg_path = os.path.join(self.graph_directory, 'HCLG.fst')
        self.lm_gz_path = os.path.join(self.lm_directory, 'lm.gz')
        os.makedirs(self.graph_directory, exist_ok=True)
        os.makedirs(self.lm_directory, exist_ok=True)
    
    def make_transducer(self, lmtext_path: str, skip: bool=False) -> None:

        if skip:
            generate_text_transducer(lmtext_path, self.words_path, self.g_text_path, skip)
        else:
            generate_text_transducer(lmtext_path, self.words_path, self.g_text_path)
        
        with open(self.log_file_path, 'a') as log_file, open(self.g_path, 'w') as g:
            compile_proc = subprocess.Popen(
                [thirdparty_binary("fstcompile"), f"--isymbols={self.words_path}",
                f"--osymbols={self.words_path}", self.g_text_path],
                    stderr=log_file,
                    stdout=subprocess.PIPE,
                    env=os.environ,
            )
            determinize_proc = subprocess.Popen(
                [thirdparty_binary("fstdeterminizestar")],
                stdin=compile_proc.stdout,
                stdout=g,
                stderr=log_file,
                env=os.environ,
            )
            minimize_proc = subprocess.Popen(
                [thirdparty_binary("fstminimize")],
                stdin=determinize_proc.stdout,
                stderr=log_file,
                # stdout=g,
                env=os.environ,

            )
            determinize_proc.communicate()
            

    def make_trigram(self, lmtext_path: str):

        with open(self.log_file_path, 'a') as log_file:
            build_proc = subprocess.Popen(
                [thirdparty_binary('build-lm.sh'), "-i", lmtext_path, "-o", self.lm_gz_path],
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )

            compile_proc = subprocess.Popen(
                [thirdparty_binary('compile-lm'), self.lm_gz_path, "-t=yes", "/dev/stdout"],
                stdin=build_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=os.environ,
            )
            to_fst_proc = subprocess.Popen(
                [thirdparty_binary("arpa2fst"), "--disambig-symbol=#0",
                f"--read-symbol-table={self.words_path}", "-", self.g_path],
                stdin=compile_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            to_fst_proc.communicate()

        



    def make_hclg(self) -> None:
        with open(self.log_file_path, 'a') as log_file:


            log_file.write("Generating LG.fst...")
            compose_lg(self.disambig_L_path, self.g_path, self.lg_path, log_file)

            context_width = get_tree_info('tree', log_file, 'context-width')
            central_pos = get_tree_info('tree', log_file, 'central-position')
            ilabels_temp = f"ilabels_{context_width}_{central_pos}"
            out_disambig = f"disambig_ilabels_{context_width}_{central_pos}"

            log_file.write("Generating CLG.fst...")
            compose_clg(
                self.disambig_int_path,
                out_disambig,
                context_width,
                central_pos,
                ilabels_temp,
                self.lg_path,
                self.clg_path,
                log_file,
            )

            compose_hclg(
                self.working_directory,
                ilabels_temp,
                self.transition_scale,
                self.clg_path,
                self.hclga_path,
                log_file,
            )


            log_file.write("Generating HCLG.fst...")
            self_loop_proc = subprocess.Popen(
                [
                    thirdparty_binary("add-self-loops"),
                    f"--self-loop-scale={self.self_loop_scale}",
                    "--reorder=true",
                    self.final_model_path,
                    self.hclga_path,
                ],
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            convert_proc = subprocess.Popen(
                [
                    thirdparty_binary("fstconvert"),
                    "--v=100",
                    "--fst_type=const",
                    "-",
                    self.hclg_path,
                ],
                stdin=self_loop_proc.stdout,
                stderr=log_file,
                env=os.environ,
            )
            convert_proc.communicate()

    def mkgraph(self, lm_text: str, type: str):

        if type == 'trigram':
            self.make_trigram(lm_text)
            self.make_hclg()
        elif type == 'transducer':
            self.make_transducer(lm_text)
            self.make_hclg()






class Transcriber():
    """
    Class for performing transcription.
    
    Parameters
    ----------

    final_model_path: str
        Path to final.mdl
    tree_path: str
        Path to tree
    hclg_path: str
        Path to HCLG.fst graph
    words_path: str
        Path to words.txt
    disambig_int_path: str 
        Path to disambiguation int symbols
        (disambig.int)
    word_boundary_int_path: str
        Path to word boundary int symbols
        (word_boundary.int)
    
    """


    def __init__(
        self,
        # final_model_path: str,
        # tree_path: str,
        # hclg_path: str,
        # words_path: str,
        # disambig_int_path: str,
        # lexicon_fst_path: str,
        # word_boundary_int_path: str
        args: TranscriberArgs 
    ): 



        self.final_model_path = args.final_model_path
        self.tree_path = args.tree_path
        self.hclg_path = args.hclg_path
        self.words_path = args.words_path
        self.disambig_int_path = args.disambig_int_path
        self.lexicon_fst_path = args.lexicon_fst_path
        self.word_boundary_int_path = args.word_boundary_int_path
        self.wb_info = WordBoundaryInfo.from_file(WordBoundaryInfoNewOpts(),
                                            self.word_boundary_int_path)


        # Construct recognizer
        self.decoder_opts = LatticeFasterDecoderOptions()
        self.decoder_opts.beam = 11.0
        self.decoder_opts.max_active = 7000
        self.asr = GmmLatticeFasterRecognizer.from_files(
            final_model, hclg, words, decoder_opts=self.decoder_opts)

    def decode(self, feats_ark):
        # Decode
        out = []
        for key, feats in SequentialMatrixReader(f'ark:{feats_ark}'):
            out.append((key, self.asr.decode(feats)))
        return out

    def decode_text(self, feats_ark):
        """
        Transcribe audio from features.

        Parameters
        ----------
        feats_ark: str
            Path to feats_ark
        
        """
        return [(key, t['text']) for key, t in self.decode(feats_ark)]



    def decode_ctm(self, feats_ark):
        """
        Transcribe and align audio features.

        Parameters
        ----------
        feats_ark: str
            Path to feats_ark
        """

        best_paths = [(key, t['best_path']) for key, t in  self.decode(feats_ark)]
        aligner = GmmAligner.from_files("gmm-boost-silence --boost=1.0 1 {} - |".format(self.final_model_path),
                                        self.tree_path, self.lexicon_fst_path, self.words_path, self.disambig_int_path,
                                        self_loop_scale=0.1)
        return [(key, [ctmEntry(word=w, onset=on, duration=dur) \
                for w, on, dur in aligner.to_word_alignment(best_path=best_path, word_boundary_info=self.wb_info) if w!= '<eps>']) \
                for key, best_path in best_paths]




class DecodeSegments():

    def __init__(
        self,
        feature_extractor: Mfcc,
        hclg: CreateHCLG,
        transcriber: Transcriber,
        reference: List[str],
        segments_dir_path: str,
        init: bool
    ) -> None:

        self.feature_extractor = feature_extractor
        self.hclg = hclg
        self.transcriber = transcriber 
        self.reference = reference
        self.segments_dir_path, segments_dir_path
        if init:
            self.create_dirs = True

    def create_segments_file(
        self,
        unaligned_region,
        segment_data_dir_path
    ) -> None:

        segments = os.path.join(segment_data_dir_path, 'segments')
        with open(segments, 'w') as f:
            f.write('segment_{onset}_{offset} key_1 {onset} {offset}'.format(
                onset=unaligned_region.onset_time,
                offset=unaligned_region.offset_time
                )
            )
        return segments

    def create_segment_text(
        self,
        unaligned_region,
        segment_data_dir_path) -> str:
        
        text_path = os.path.join(segment_data_dir_path, 'text')
        text = self.reference[unaligned_region.onset_index : unaligned_region.offset_index+1]
        with open(text, 'w') as f:
            f.write(' '.join(text))
        return text_path


    def decode_segment(
        self,
        segment,
        feats_ark,
        text

    ) -> None:
        self.feature_extractor.make_feats(segment_path=segment)
        self.hclg.mkgraph(text, 'trigram')
        hypothesis_ctm = self.transcriber.decode_ctm(feats_ark)[0][1]
        hypothesis = self.transcriber.decode_text(feats_ark)[0][1]

        return hypothesis, hypothesis_ctm


        