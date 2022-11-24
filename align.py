from transcriber import CreateHCLG, Transcriber
from kaldialign import align
from features import Mfcc
from utils import read_reference_text, create_hclg_args, create_transcriber_args, create_mfcc_args
from alignment import T2TAlignment

disambig_L_path = 'model/L_disambig.fst'
words_path = 'model/words.txt'
disambig_int_path = 'model/disambig.int'
working_directory = 'working_dir'
model_path='model/final.mdl'
tree_path='model/tree'
lexicon_fst_path='model/L.fst'
word_boundary_int_path='model/word_boundary.int'
final_matrix = 'model/final.mat'

reference_path='p120032.d_word_3.txt'

hclg_args = create_hclg_args('model', 'working_dir')
transcriber_args = create_transcriber_args('model', 'working_dir')
mfcc_args = create_mfcc_args('model', 'working_dir')
hclg = CreateHCLG(hclg_args)
transcriber = Transcriber(transcriber_args)
mfcc = Mfcc(mfcc_args)
mfcc.make_feats(segment_path='working_dir/segments')



hclg.mkgraph(reference_path, 'trigram')

hypothesis_ctm = transcriber.decode_ctm('working_dir/feats.ark')[0][1]
hypothesis = transcriber.decode_text('working_dir/feats.ark')[0][1]
hypothesis = hypothesis.split()
reference = read_reference_text(reference_path)



t2talignment = T2TAlignment()

alignment_lab = t2talignment.run(
        reference=reference,
        hypothesis=hypothesis,
        hypothesis_ctm=hypothesis_ctm,
        current_alignment = None,
        text_onset_index=0,
        segment_onset_time=.0
)


print(alignment_lab)