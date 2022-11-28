import os
from aligner.transcriber import CreateHCLG, Transcriber, DecodeSegments
from kaldialign import align
from aligner.features import Mfcc
from aligner.utils import arg_parser, read_reference_text, initialize_working_dir, read_yaml_file
from aligner.config import create_transcriber_args, create_mfcc_args, create_hclg_args
from aligner.alignment import T2TAlignment
from aligner.segmenter import Segmenter


config = read_yaml_file('settings.yaml')
parser = arg_parser()
args = parser.parse_args()
audio_path = args.a
transcription_path= args.t
model_dir_path = args.m
working_dir_path = args.w
vad_access_token = config['vad']['vad_access_token']
vad_duration = config['vad']['duration']

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
mfcc_args = create_mfcc_args(model_dir_path, working_dir_path)
hclg = CreateHCLG(hclg_args)
mfcc = Mfcc(mfcc_args)


# segmenter = Segmenter(access_token=vad_access_token, segment_duration=15)
# segmenter.run(working_dir_path, audio_path)

mfcc.make_feats(segment_path=init_segment)
# exit(1)
hclg.mkgraph(lm_text, 'trigram')

transcriber = Transcriber(transcriber_args)
print(transcriber.decode_text(init_feats_ark))
hypothesis_ctm = transcriber.decode_ctm(init_feats_ark)[0][1]
hypothesis = transcriber.decode_text(init_feats_ark)[0][1]
hypothesis = hypothesis.split()
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

with open('greek_text.lab', 'w') as f:
        for entry in current_alignment:
                f.write(f'{entry.onset} {entry.offset} {entry.word}\n')

segments_function = DecodeSegments(
        model_dir=model_dir_path,
        wav_scp=os.path.join(working_dir_path, 'wav.scp'),
        feature_extractor=mfcc,
        reference=reference,
        segments_dir_path=segments_data_dirs,
        init = True
)




unaligned_regions_hypothesis = segments_function.decode_parallel(unaligned_regions)
for iter in range(3):
        for segment_data in unaligned_regions_hypothesis:
                reference = read_reference_text(f'working_dir/segments_data/{segment_data.segment_name}/text')

                current_alignment, unaligned_regions = t2talignment.run(
                reference=reference,
                hypothesis=segment_data.hypothesis.split(),
                hypothesis_ctm=segment_data.hypothesis_ctm,
                current_alignment = current_alignment,
                text_onset_index=segment_data.onset_index,
                segment_onset_time=segment_data.onset_time
        )

        with open('greek_text_iter{iter}.lab', 'w') as f:
                for entry in current_alignment:
                        f.write(f'{entry.onset} {entry.offset} {entry.word}\n')

