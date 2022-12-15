import typing
import os
import shutil
import subprocess
from typing import List, Tuple, Dict, Optional
from multiprocessing import Process, Queue
from kaldi.asr import GmmLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.util.table import SequentialMatrixReader
from kaldi.lat.align import WordBoundaryInfoNewOpts, WordBoundaryInfo
from kaldi.alignment import GmmAligner
from aligner.graph import compose_clg, compose_hclg, compose_lg, generate_text_transducer, get_tree_info
from aligner.utils import thirdparty_binary, ctmEntry, UnaliRegion, SegmentHypothesis
from aligner.config import create_hclg_args, create_mfcc_args, create_transcriber_args, create_fmllr_args, \
    HCLGArgs, TranscriberArgs
from aligner.features import Mfcc
from aligner.fmllr_decoding import Transcriber



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

):
        self.model_dir_path = args.model_dir_path
        self.tree_path = os.path.join(self.model_dir_path, 'tree')
        self.disambig_L_path = args.disambig_L_path
        self.disambig_int_path = args.disambig_int_path
        self.working_directory = args.working_directory
        self.log_file_path = args.log_file_path
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
        
        with open(self.log_file_path, 'a') as log_file:
            with open(self.g_path, 'w') as g:
                # compile_proc = subprocess.Popen(
                #     [thirdparty_binary("fstcompile"), f"--isymbols={self.words_path}",
                #     f"--osymbols={self.words_path}", self.g_text_path],
                #         stderr=log_file,
                #         stdout=subprocess.PIPE,
                #         env=os.environ,
                # )
                # determinize_proc = subprocess.Popen(
                #     [thirdparty_binary("fstdeterminizestar")],
                #     stdin=compile_proc.stdout,
                #     stdout=subprocess.PIPE,
                #     stderr=log_file,
                #     env=os.environ,
                # )
                # subprocess.Popen(
                #     [thirdparty_binary("fstminimize")],
                #     stdin=determinize_proc.stdout,
                #     stderr=log_file,
                #     stdout=g,
                #     env=os.environ,

                # )
                # determinize_proc.communicate()

                # print(f'{thirdparty_binary("fstcompile")} --isymbols={self.words_path} --osymbols={self.words_path} {self.g_text_path} | fstdeterminizestar | fstminimize> {self.g_path}')
                subprocess.call(f'{thirdparty_binary("fstcompile")} --isymbols={self.words_path} --osymbols={self.words_path} {self.g_text_path} | fstdeterminizestar | fstminimize> {self.g_path}', stderr=log_file, shell=True)
            

    def make_trigram(self, lmtext_path: str):
        with open(self.log_file_path, 'a') as log_file:
            subprocess.run(
                [thirdparty_binary('build-lm.sh'), "-i", lmtext_path, "-o", self.lm_gz_path],
                stderr=log_file,
                stdout=subprocess.PIPE,
                env=os.environ,
            )
            compile_proc = subprocess.Popen(
                [thirdparty_binary('compile-lm'), self.lm_gz_path, "-t=yes", "/dev/stdout"],
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
            # with open(self.g_path, 'w') as g:
            #     determinize_proc = subprocess.Popen(
            #         [thirdparty_binary("fstdeterminizestar"), self.g_path],
            #         stdout=subprocess.PIPE,
            #         stderr=log_file,
            #         env=os.environ,
            #     )
            #     minimize_proc = subprocess.Popen(
            #         [thirdparty_binary("fstminimize")],
            #         stdin=determinize_proc.stdout,
            #         stderr=log_file,
            #         stdout=g,
            #         env=os.environ,
            #     )
            #     minimize_proc.communicate()

            

        



    def make_hclg(self) -> None:
        with open(self.log_file_path, 'a') as log_file:


            log_file.write("Generating LG.fst...\n")
            compose_lg(self.disambig_L_path, self.g_path, self.lg_path, log_file)

            context_width = get_tree_info(self.tree_path, log_file, 'context-width')
            central_pos = get_tree_info(self.tree_path, log_file, 'central-position')
            ilabels_temp = os.path.join(
                self.graph_directory,
                f"ilabels_{context_width}_{central_pos}"
            )
            out_disambig = os.path.join(
                self.graph_directory,
                f"disambig_ilabels_{context_width}_{central_pos}",
            )                


            log_file.write("Generating CLG.fst...\n")
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
                self.model_dir_path,
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






# class Transcriber():
#     """
#     Class for performing transcription.
    
#     Parameters
#     ----------

#     final_model_path: str
#         Path to final.mdl
#     tree_path: str
#         Path to tree
#     hclg_path: str
#         Path to HCLG.fst graph
#     words_path: str
#         Path to words.txt
#     disambig_int_path: str 
#         Path to disambiguation int symbols
#         (disambig.int)
#     word_boundary_int_path: str
#         Path to word boundary int symbols
#         (word_boundary.int)
    
#     """


#     def __init__(
#         self,
#         args: TranscriberArgs 
#     ): 



#         self.final_model_path = args.final_model_path
#         self.tree_path = args.tree_path
#         self.hclg_path = args.hclg_path
#         self.words_path = args.words_path
#         self.disambig_int_path = args.disambig_int_path
#         self.lexicon_fst_path = args.lexicon_fst_path
#         self.word_boundary_int_path = args.word_boundary_int_path
#         self.wb_info = WordBoundaryInfo.from_file(WordBoundaryInfoNewOpts(),
#                                             self.word_boundary_int_path)


#         # Construct recognizer
#         self.decoder_opts = LatticeFasterDecoderOptions()
#         self.decoder_opts.beam = 11.0
#         self.decoder_opts.max_active = 7000

#         self.asr = GmmLatticeFasterRecognizer.from_files(
#             self.final_model_path,
#             self.hclg_path,
#             self.words_path,
#             decoder_opts=self.decoder_opts)

#     def decode(self, feats_ark):
#         out = []
#         for key, feats in SequentialMatrixReader(f'ark:{feats_ark}'):
#             out.append((key, self.asr.decode(feats)))
#         return out

#     # def decode_text(self, feats_ark):
#     #     """
#     #     Transcribe audio from features.

#     #     Parameters
#     #     ----------
#     #     feats_ark: str
#     #         Path to feats_ark
        
#     #     """
#     #     return [(key, t['text']) for key, t in self.decode(feats_ark)]

    # def fix_ctm_timing(self, hypothesis_ctm):

    #     hypothesis_ctm_fixed = []
    #     for key, segment_ctm in hypothesis_ctm:
    #         segment_onset_time = round(float(key.split("_")[1]) * 100)
    #         hypothesis_ctm_fixed+=[ctmEntry( 
    #             word=entry.word,
    #             onset=entry.onset + segment_onset_time,
    #             duration=entry.duration
    #             ) for entry in segment_ctm]
    #     return hypothesis_ctm_fixed

#     def decode_ctm(self, feats_ark):
#         """
#         Transcribe and align audio features.

#         Parameters
#         ----------
#         feats_ark: str
#             Path to feats_ark
#         """
#         hypothesis = [hyp for hyps in [t['text'].split() for _, t in self.decode(feats_ark)] for hyp in hyps]
#         best_paths = [(key, t['best_path']) for key, t in  self.decode(feats_ark)]
#         aligner = GmmAligner.from_files("gmm-boost-silence --boost=1.0 1 {} - |".format(self.final_model_path),
#                                         self.tree_path, self.lexicon_fst_path, self.words_path, self.disambig_int_path,
#                                         self_loop_scale=0.1)
        
#         hypothesis_ctm =  [(key, [ctmEntry(word=w, onset=on, duration=dur) \
#                 for w, on, dur in aligner.to_word_alignment(best_path=best_path, word_boundary_info=self.wb_info) if w!= '<eps>']) \
#                 for key, best_path in best_paths]
#         hypothesis_ctm = self.fix_ctm_timing(hypothesis_ctm)
#         # hypothesis_ctm = [hyp for hyps in hypothesis_ctm for hyp in hyps]
        
#         return hypothesis, hypothesis_ctm



class DecodeSegments():

    def __init__(
        self,
        model_dir: str,
        wav_scp: str,
        feature_extractor: Mfcc,
        reference: List[str],
        segments_dir_path: str,
        config,
        init: bool
    ) -> None:
        self.model_dir = model_dir
        self.wav_scp = wav_scp
        self.feature_extractor = feature_extractor
        self.reference = reference
        self.segments_dir_path = segments_dir_path
        if init:
            self.create_dirs = True

        self.segment_results = Queue()
        self.config = config

    def create_segments_file(
        self,
        unaligned_region: List[UnaliRegion],
        segment_data_dir_path: str
    ) -> None:

        segments = os.path.join(segment_data_dir_path, 'segments')
        with open(segments, 'w') as f:
            f.write('segment_{onset}_{offset} key_1 {onset} {offset}'.format(
                onset=unaligned_region.onset_time,
                offset=unaligned_region.offset_time
                )
            )
        return segments

    def create_utt2spk_spk2utt(
        self,
        unaligned_region: List[UnaliRegion],
        segment_data_dir_path: str
    ) -> None:

        utt2spk = os.path.join(segment_data_dir_path, 'utt2spk')
        spk2utt = os.path.join(segment_data_dir_path, 'spk2utt')
        text = 'segment_{onset}_{offset} segment_{onset}_{offset}\n'.format(
            onset=unaligned_region.onset_time,
            offset=unaligned_region.offset_time
            )
        with open(utt2spk, 'w') as u2s, open(spk2utt, 'w') as s2u:
            u2s.write(text)
            s2u.write(text)


    def create_segment_text(
        self,
        unaligned_region: List[UnaliRegion],
        segment_data_dir_path: str) -> Tuple[str, str]:
        
        text_path = os.path.join(segment_data_dir_path, 'text')
        text = self.reference[unaligned_region.onset_index : unaligned_region.offset_index+1]
        text = ' '.join(text)
        with open(text_path, 'w') as f:
            f.write(text)
        return text, text_path

    def create_segnemt_lm_text(
        self,
        segment_data_dir_path: str,
        text: str
    ) -> str:

        lm_text_path = os.path.join(
            segment_data_dir_path,
            'lm.txt')
        with open(lm_text_path, 'w') as f:
            f.write(f'<s> {text} </s>')
        return lm_text_path



    def prepare_segment_data_dir(
        self,
        unaligned_region: List[UnaliRegion],
        lattice_type: str

    ) -> None:
        segment_data_dir_path = os.path.join(
            self.segments_dir_path,
            f'{unaligned_region.onset_time}_{unaligned_region.offset_time}'
    )
        if os.path.exists(segment_data_dir_path):
            shutil.rmtree(segment_data_dir_path)
        os.makedirs(segment_data_dir_path, exist_ok=True)


        segments_path = self.create_segments_file(
            segment_data_dir_path=segment_data_dir_path,
            unaligned_region=unaligned_region)
        text, text_path = self.create_segment_text(
            segment_data_dir_path=segment_data_dir_path,
            unaligned_region=unaligned_region
        )
        lm_text_path = self.create_segnemt_lm_text(
            segment_data_dir_path=segment_data_dir_path,
            text=text
        )
        self.create_utt2spk_spk2utt(
            segment_data_dir_path=segment_data_dir_path,
            unaligned_region=unaligned_region
        )


        hclg_args = create_hclg_args(
            self.model_dir,
            segment_data_dir_path)
        hclg = CreateHCLG(hclg_args)

        if lattice_type=='transducer':
            hclg.mkgraph(text_path, lattice_type)
        else:
            hclg.mkgraph(lm_text_path, lattice_type)


        self.feature_extractor.make_feats(
            segment_path=segments_path)
        
        shutil.copyfile(
            self.wav_scp,
            os.path.join(segment_data_dir_path, 'wav.scp')
            )
        mfcc_args = create_mfcc_args(
            self.model_dir,
            segment_data_dir_path)
        mfcc = Mfcc(mfcc_args)
        mfcc.make_feats(segment_path=segments_path)


        fmllr_args = create_fmllr_args(
            working_dir=segment_data_dir_path,
            model_dir_path=self.model_dir,
            config=self.config
            )
        transcriber = Transcriber(fmllr_args)

        return segment_data_dir_path, transcriber

    def queue_dump(self, queue):
        elements = []
        while queue.qsize():
            elements.append(queue.get())
        return elements


    def decode_segment(
        self,
        unaligned_region,
        lattice_type: str

    ) -> None:

        segment_data_dir_path, transcriber = self.prepare_segment_data_dir(
                unaligned_region=unaligned_region,
                lattice_type = lattice_type
            )   
        hypothesis, hypothesis_ctm =transcriber.decode(segment_data_dir_path)
        if hypothesis == []:
            self.segment_results.put(None)
        else:
            self.segment_results.put(
                SegmentHypothesis(
                    segment_name=f'{unaligned_region.onset_time}_{unaligned_region.offset_time}',
                    onset_index = unaligned_region.onset_index,
                    onset_time = unaligned_region.onset_time,
                    hypothesis=hypothesis,
                    hypothesis_ctm=hypothesis_ctm
                    
                )
            )

    def decode_parallel(
        self,
        unaligned_regions,
        lattice_type: str

    ) -> None:

        decode_procs = []
        for unaligned_region in unaligned_regions:
            # self.decode_segment(unaligned_region, lattice_type)
            decode_proc = Process(
                target=self.decode_segment,
                args = (unaligned_region, lattice_type)
            )
            decode_procs.append(decode_proc)
            decode_proc.start()
        for decode_proc in decode_procs:
            decode_proc.join()

            # break
        return self.queue_dump(self.segment_results)





        