import os
from pyannote.audio import Pipeline
from aligner.utils import VADSegement
from typing import List

class Segmenter():

    def __init__(
        self,
        access_token: str,
        segment_duration: int
    ) -> None:
        self.access_token = access_token
        self.segment_duration = segment_duration
    



    def run_vad(
        self,
        audio_path: str
        
    ) -> List[VADSegement]:

        pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=self.access_token)

        output = pipeline(audio_path)
        
        vad_segments = [VADSegement(
            onset_time=speech.start,
            offset_time=speech.end
        ) for speech in output.get_timeline().support()]

        return vad_segments

    def create_segments_from_vad(
        self,
        data_dir_path: str,
        vad_segments: List[VADSegement]
    ) -> None:

        segments_path = os.path.join(
            data_dir_path,
            'segments'
        )
        segment_onset = vad_segments[0].onset_time
        duration = vad_segments[0].offset_time
        with open(segments_path, 'w') as fd:
            for idx, vad_segment in enumerate(vad_segments[1:]):
                
                if duration <= self.segment_duration:
                    duration = vad_segment.offset_time - segment_onset
                else:
                    fd.write('segment_{on}_{off} key_1 {on} {off}\n'.format(
                        on=segment_onset,
                        off=vad_segments[idx].offset_time
                    ))
                    segment_onset = vad_segment.onset_time
                    duration = vad_segment.offset_time - segment_onset
            fd.write('segment_{on}_{off} key_1 {on} {off}\n'.format(
                on=segment_onset,
                off=vad_segments[-1].offset_time
            ))
    
    def create_utt2spk(
        self,
        data_dir
        ) -> None:
        segments_path = os.path.join(data_dir, 'segments')
        utt2spk_path = os.path.join(data_dir, 'utt2spk')
        spk2utt_path = os.path.join(data_dir, 'spk2utt')
        with open(segments_path, 'r') as fd:
            segment_names = [ln.split()[0] for ln in fd]
        with open(utt2spk_path, 'w') as u2s, open(spk2utt_path, 'w') as s2u:
            for segment_name in segment_names:
                u2s.write(f"{segment_name} {segment_name}\n")
                s2u.write(f"{segment_name} {segment_name}\n")

    def sort_segments(
        self,
        data_dir
    ) -> None:
        segments_path = os.path.join(data_dir, 'segments')
        with open(segments_path, 'r') as fd:
            segments = [ln for ln in fd]
        segments_kaldi_sorted = sorted(segments)
        with open(segments_path, 'w') as fd:
            for segment in segments_kaldi_sorted:
                fd.write(segment)

    def run(
        self,
        data_dir_path: str,
        audio_path: str
      ) -> None:

        vad_segments = self.run_vad(audio_path)
        self.create_segments_from_vad(data_dir_path, vad_segments)
        self.sort_segments(data_dir_path)
        self.create_utt2spk(data_dir_path)
