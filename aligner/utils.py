import shutil
import os
import yaml
import argparse
import sys
from dataclasses import dataclass
from typing import List, Dict, Any
from yaml.loader import SafeLoader
from loguru import logger



MetaDict = Dict[str, Any]

@dataclass
class ctmEntry:
    word: str
    onset: int
    duration: int

@dataclass
class labEntry:
    word: str
    onset: float
    offset: float

@dataclass
class IslandSegment:
    onset_index: int
    offset_index: int

@dataclass
class UnaliRegion:
    onset_index: int
    offset_index: int
    onset_time: float
    offset_time: float

@dataclass
class SegmentHypothesis:
    segment_name: str
    hypothesis: List[str]
    onset_index: int
    onset_time: float
    hypothesis_ctm: List[ctmEntry]

@dataclass
class VADSegement:
    onset_time: float
    offset_time: float

    def duration(self) -> float:
        return self.offset_time - self.onset_time


def thirdparty_binary(binary_name: str) -> str:
    """
    Generate full path to a given binary name
    Notes
    -----
    With the move to conda, this function is deprecated as conda will manage the path much better
    Parameters
    ----------
    binary_name: str
        Executable to run
    Returns
    -------
    str
        Full path to the executable
    """
    bin_path = shutil.which(binary_name)
    if bin_path is None:
        if binary_name in ["fstcompile", "fstarcsort", "fstconvert"]:
            raise Exception("Install Openfst and add it to path.")
        elif binary_name in ["build-lm.sh"]:
            raise Exception("Install IRSTLM and add it to path.")
        else:
            raise Exception(f"{binary_name} not in path")
    if " " in bin_path:
        return f'"{bin_path}"'
    return bin_path



def read_reference_text(ref_path: str) -> List[str]:
    """
    Parameters
    ----------
    ref_path: str
        Path to refrence text
    Returns
    -------
    List[str]
        List of words in refrence text
    """
    with open(ref_path, 'r') as fd:
        return fd.readline().rstrip().split()



def read_yaml_file(yaml_path):
    with open(yaml_path, 'r') as f:
        data = list(yaml.load_all(f, Loader=SafeLoader))
        return data[0]



def initialize_working_dir(
    audio_path: str,
    transcription_path: str, 
    working_dir_path: str,
    config
    ) -> None:

    segments_data_dirs = os.path.join(
            working_dir_path,
            config['working_dir_paths']['segments_data']
    )
    lm_text_path = os.path.join(
            working_dir_path,
            config['working_dir_paths']['lm_text']
    )

    os.makedirs(working_dir_path, exist_ok=True)
    graph_directory = os.path.join(working_dir_path, 'graph_dir')
    lm_directory = os.path.join(working_dir_path, 'lm_dir')
    os.makedirs(graph_directory, exist_ok=True)
    os.makedirs(lm_directory, exist_ok=True)
    os.makedirs(segments_data_dirs, exist_ok=True)
    # create wav.scp
    wav_scp = os.path.join(working_dir_path, 'wav.scp')
    with open(wav_scp, 'w') as f:
        f.write(f'key_1 {audio_path}')
    
    # create segments
    segments = os.path.join(working_dir_path, 'segments')
    with open(segments, 'w') as f:
        f.write('key_1 key_1 0 -1')

    # create spk2utt
    segments = os.path.join(working_dir_path, 'spk2utt')
    with open(segments, 'w') as f:
        f.write('key_1 key_1\n')

    # create utt2spk
    segments = os.path.join(working_dir_path, 'utt2spk')
    with open(segments, 'w') as f:
        f.write('key_1 key_1\n')



    # create lm text
    with open(lm_text_path, 'w') as lm:
        with open(transcription_path, 'r') as transc:
            reference_text = transc.readline().rstrip()
        lm.write(f'<s> {reference_text} </s>')

    
def arg_parser():

    parser = argparse.ArgumentParser(description="Speech to text alignmener for long audio")
    parser.add_argument('-a', type=str, help="Path to audio file")
    parser.add_argument("-t", type=str, help="Path to transcription")
    parser.add_argument("-m", type=str, help="Path to model directory")
    parser.add_argument('-w', type=str, help="Path to working directory")

    return parser



class AlignmentLogger():

    def __init__(
        self,
        working_dir_path,
        config
    ) -> None:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add(config['working_dir_paths']['log_file'])
        self.working_dir_path = working_dir_path
        self.segments_data_dir = os.path.join(
            self.working_dir_path,
            config['working_dir_paths']['segments_data'])


    def dump_log_to_main(
        self,
        log_path,
        header=''
        ) -> None:
        
        if header != '':
            logger.debug(header)
        with open(log_path, 'r') as log:
            for ln in log:
                logger.debug(ln)



    def append_decoding_logs(
        self,
    ) -> None:

        for segment_data_dir_path in os.listdir(self.segments_data_dir):
            decoding_log_path = os.path.join(
                self.segments_data_dir,
                segment_data_dir_path,
                self.config['working_dir_paths']['decoding.log']
                )
            header = f'Decoding Log for segment {segment_data_dir_path}'
            self.dump_log(decoding_log_path, header=header)  

            
