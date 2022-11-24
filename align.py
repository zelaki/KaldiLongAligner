import os
from transcriber import CreateHCLG, Transcriber
from kaldialign import align
from features import Mfcc
from utils import arg_parser, read_reference_text, \
        create_hclg_args, create_transcriber_args, \
        create_mfcc_args, initialize_working_dir, read_yaml_file
from alignment import T2TAlignment



config = read_yaml_file('settings.yaml')
parser = arg_parser()
args = parser.parse_args()
audio_path = args.a
transcription_path= args.t
model_dir_path = args.m
working_dir_path = args.w
init_segment = os.path.join(
        working_dir_path,
        config['working_dir_paths']['segments']
)
init_feats_ark = os.path.join(
        working_dir_path,
        config['working_dir_paths']['feats_ark']
)
initialize_working_dir(
        audio_path,
        transcription_path,
        working_dir_path)

hclg_args = create_hclg_args(model_dir_path, working_dir_path)
transcriber_args = create_transcriber_args(model_dir_path, working_dir_path)
mfcc_args = create_mfcc_args(model_dir_path, working_dir_path)
hclg = CreateHCLG(hclg_args)
transcriber = Transcriber(transcriber_args)
mfcc = Mfcc(mfcc_args)
mfcc.make_feats(segment_path=init_segment)



hclg.mkgraph(transcription_path, 'trigram')
hypothesis_ctm = transcriber.decode_ctm(init_feats_ark)[0][1]
hypothesis = transcriber.decode_text(init_feats_ark)[0][1]
hypothesis = hypothesis.split()
reference = read_reference_text(transcription_path)



t2talignment = T2TAlignment()

alignment_lab, unaligned_regions = t2talignment.run(
        reference=reference,
        hypothesis=hypothesis,
        hypothesis_ctm=hypothesis_ctm,
        current_alignment = None,
        text_onset_index=0,
        segment_onset_time=.0
)



print(unaligned_regions)



