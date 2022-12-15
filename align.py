import os
from aligner.transcriber import CreateHCLG, DecodeSegments
from kaldialign import align
from aligner.features import Mfcc
from aligner.utils import AlignmentLogger, arg_parser, read_reference_text, initialize_working_dir, read_yaml_file
from aligner.config import create_transcriber_args, create_mfcc_args, create_hclg_args, create_fmllr_args
from aligner.alignment import T2TAlignment
from aligner.segmenter import Segmenter
from aligner.fmllr_decoding import Transcriber

config = read_yaml_file('settings.yaml')
parser = arg_parser()
args = parser.parse_args()
audio_path = args.a
transcription_path= args.t
model_dir_path = args.m
working_dir_path = args.w
vad_access_token = config['vad']['vad_access_token']
vad_duration = config['vad']['duration']
alignment_logger = AlignmentLogger(working_dir_path=working_dir_path, config=config)
init_segment = os.path.join(
working_dir_path,
config['working_dir_paths']['segments']
)
init_feats_ark = os.path.join(
        working_dir_path,
        config['working_dir_paths']['feats_ark']
)
lm_text = os.path.join(
        working_dir_path,
        config['working_dir_paths']['lm_text']
)
segments_data_dirs = os.path.join(
        working_dir_path,
        config['working_dir_paths']['segments_data']
)
initialize_working_dir(
        audio_path,
        transcription_path,
        working_dir_path,
        config)

hclg_args = create_hclg_args(model_dir_path, working_dir_path)
transcriber_args = create_transcriber_args(model_dir_path, working_dir_path)
fmllr_args = create_fmllr_args(working_dir_path, model_dir_path, config)
mfcc_args = create_mfcc_args(model_dir_path, working_dir_path)
hclg = CreateHCLG(hclg_args)
mfcc = Mfcc(mfcc_args)


segmenter = Segmenter(access_token=vad_access_token, segment_duration=15)
segmenter.run(working_dir_path, audio_path)
mfcc.make_feats(segment_path=init_segment)
hclg.mkgraph(lm_text, 'trigram')



# transcriber = Transcriber(transcriber_args)
transcriber = Transcriber(fmllr_args)
hypothesis, hypothesis_ctm = transcriber.decode(working_dir_path)
print(hypothesis)
# exit(1)
reference = read_reference_text(transcription_path)

t2talignment = T2TAlignment()
current_alignment, unaligned_regions = t2talignment.run(
        reference=reference,
        hypothesis=hypothesis,
        hypothesis_ctm=hypothesis_ctm,
        current_alignment = None,
        text_onset_index=0,
        segment_onset_time=.0
)



with open('alignment.lab', 'w') as f:
        for entry in current_alignment:
                f.write(f'{entry.onset} {entry.offset} {entry.word}\n')



exit(1)


segments_function = DecodeSegments(
        model_dir=model_dir_path,
        wav_scp=os.path.join(working_dir_path, 'wav.scp'),
        feature_extractor=mfcc,
        reference=reference,
        segments_dir_path=segments_data_dirs,
        init = True,
        config=config
)


lattice_type = 'trigram'
unaligned_regions_hypothesis = segments_function.decode_parallel(unaligned_regions, lattice_type)
for iter in range(3):
        for segment_data in unaligned_regions_hypothesis:
                if segment_data==None: continue
                reference = read_reference_text(f'working_dir/segments_data/{segment_data.segment_name}/text')

                current_alignment, unaligned_regions = t2talignment.run(
                reference=reference,
                hypothesis=segment_data.hypothesis,
                hypothesis_ctm=segment_data.hypothesis_ctm,
                current_alignment = current_alignment,
                text_onset_index=segment_data.onset_index,
                segment_onset_time=segment_data.onset_time
        )
        if iter >= 1:
                lattice_type = 'transducer'

        unaligned_regions_hypothesis = segments_function.decode_parallel(unaligned_regions, lattice_type)


        with open(f'alignment{iter}.lab', 'w') as f:
                for entry in current_alignment:
                        f.write(f'{entry.onset} {entry.offset} {entry.word}\n')

