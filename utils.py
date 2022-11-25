import shutil
import os
import yaml
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
from yaml.loader import SafeLoader


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


@dataclass
class HCLGArgs():
    disambig_L_path: str
    disambig_int_path: str
    working_directory: str
    final_model_path: str 
    words_path: str
    model_dir_path: str

@dataclass
class TranscriberArgs():
    final_model_path: str
    tree_path: str
    hclg_path: str
    words_path: str
    disambig_int_path: str
    lexicon_fst_path: str 
    word_boundary_int_path: str 

@dataclass
class MFCCArgs():
    wav_path: str
    feats_scp_path: str 
    mfcc_options: MetaDict
    pitch_options: MetaDict
    final_matrix: str
    log_path: str

def create_mfcc_args(model_dir_path, working_dir_path):

    features_config = FeatureConfigMixin()
    args = MFCCArgs(
        wav_path = os.path.join(working_dir_path, 'wav.scp'),
        feats_scp_path = os.path.join(working_dir_path, 'feats.scp'),
        mfcc_options = features_config.mfcc_options,
        pitch_options = features_config.pitch_options,
        final_matrix=os.path.join(model_dir_path, 'final.mat'),
        log_path=os.path.join(working_dir_path, 'log.txt')
    )
    return args 


def create_transcriber_args(model_dir_path: str, working_dir_path: str) -> TranscriberArgs:
    args = TranscriberArgs(
        final_model_path = os.path.join(model_dir_path, 'final.mdl'),
        tree_path = os.path.join(model_dir_path, 'tree'),
        hclg_path = os.path.join(working_dir_path, 'graph_dir', 'HCLG.fst'),
        words_path = os.path.join(model_dir_path, 'words.txt'),
        disambig_int_path = os.path.join(model_dir_path, 'disambig.int'),
        lexicon_fst_path = os.path.join(model_dir_path, "L.fst"),
        word_boundary_int_path = os.path.join(model_dir_path, 'word_boundary.int')
    )
    return args

def create_hclg_args(model_dir_path: str, working_dir_path: str) -> HCLGArgs:
    args = HCLGArgs(
        disambig_L_path=os.path.join(model_dir_path, 'L_disambig.fst'),
        disambig_int_path=os.path.join(model_dir_path, 'disambig.int'),
        working_directory=working_dir_path,
        final_model_path=os.path.join(model_dir_path,'final.mdl'),
        words_path=os.path.join(model_dir_path, 'words.txt'),
        model_dir_path = model_dir_path
    )
    return args

def read_yaml_file(yaml_path):
    with open(yaml_path, 'r') as f:
        data = list(yaml.load_all(f, Loader=SafeLoader))
        return data[0]



def initialize_working_dir(
    audio_path: str,
    transcription_path: str, 
    working_dir_path: str,
    segments_data_dirs: str,
    lm_text_path: str
    ) -> None:

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


class FeatureConfigMixin:
    """
    Class to store configuration information about MFCC generation
    Attributes
    ----------
    feature_type : str
        Feature type, defaults to "mfcc"
    use_energy : bool
        Flag for whether first coefficient should be used, defaults to False
    frame_shift : int
        number of milliseconds between frames, defaults to 10
    snip_edges : bool
        Flag for enabling Kaldi's snip edges, should be better time precision
    use_pitch : bool
        Flag for including pitch in features, defaults to False
    low_frequency : int
        Frequency floor
    high_frequency : int
        Frequency ceiling
    sample_frequency : int
        Sampling frequency
    allow_downsample : bool
        Flag for whether to allow downsampling, default is True
    allow_upsample : bool
        Flag for whether to allow upsampling, default is True
    speaker_independent : bool
        Flag for whether features are speaker independent, default is True
    uses_cmvn : bool
        Flag for whether to use CMVN, default is True
    uses_deltas : bool
        Flag for whether to use delta features, default is True
    uses_splices : bool
        Flag for whether to use splices and LDA transformations, default is False
    uses_speaker_adaptation : bool
        Flag for whether to use speaker adaptation, default is False
    fmllr_update_type : str
        Type of fMLLR estimation, defaults to "full"
    silence_weight : float
        Weight of silence in calculating LDA or fMLLR
    splice_left_context : int or None
        Number of frames to splice on the left for calculating LDA
    splice_right_context : int or None
        Number of frames to splice on the right for calculating LDA
    """

    def __init__(
        self,
        feature_type: str = "mfcc",
        use_energy: bool = False,
        frame_shift: int = 10,
        frame_length: int = 25,
        snip_edges: bool = True,
        low_frequency: int = 20,
        high_frequency: int = 7800,
        sample_frequency: int = 16000,
        allow_downsample: bool = True,
        allow_upsample: bool = True,
        speaker_independent: bool = True,
        uses_cmvn: bool = True,
        uses_deltas: bool = True,
        uses_splices: bool = False,
        uses_voiced: bool = False,
        uses_speaker_adaptation: bool = False,
        fmllr_update_type: str = "full",
        silence_weight: float = 0.0,
        splice_left_context: int = 3,
        splice_right_context: int = 3,
        use_pitch: bool = False,
        min_f0: float = 50,
        max_f0: float = 500,
        delta_pitch: float = 0.005,
        penalty_factor: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_type = feature_type
        self.use_energy = use_energy
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.snip_edges = snip_edges
        self.low_frequency = low_frequency
        self.high_frequency = high_frequency
        self.sample_frequency = sample_frequency
        self.allow_downsample = allow_downsample
        self.allow_upsample = allow_upsample
        self.speaker_independent = speaker_independent
        self.uses_cmvn = uses_cmvn
        self.uses_deltas = uses_deltas
        self.uses_splices = uses_splices
        self.uses_voiced = uses_voiced
        self.uses_speaker_adaptation = uses_speaker_adaptation
        self.fmllr_update_type = fmllr_update_type
        self.silence_weight = silence_weight
        self.splice_left_context = splice_left_context
        self.splice_right_context = splice_right_context

        # Pitch features
        self.use_pitch = use_pitch
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.delta_pitch = delta_pitch
        self.penalty_factor = penalty_factor

    @property
    def vad_options(self) -> MetaDict:
        """Abstract method for VAD options"""
        raise NotImplementedError

    @property
    def alignment_model_path(self) -> str:  # needed for fmllr
        """Abstract method for alignment model path"""
        raise NotImplementedError

    @property
    def model_path(self) -> str:  # needed for fmllr
        """Abstract method for model path"""
        raise NotImplementedError

    @property
    def working_directory(self) -> str:
        """Abstract method for working directory"""
        ...

    @property
    def corpus_output_directory(self) -> str:
        """Abstract method for working directory of corpus"""
        ...

    @property
    def data_directory(self) -> str:
        """Abstract method for corpus data directory"""
        ...

    @property
    def feature_options(self) -> MetaDict:
        """Parameters for feature generation"""
        options = {
            "type": self.feature_type,
            "use_energy": self.use_energy,
            "frame_shift": self.frame_shift,
            "frame_length": self.frame_length,
            "snip_edges": self.snip_edges,
            "low_frequency": self.low_frequency,
            "high_frequency": self.high_frequency,
            "sample_frequency": self.sample_frequency,
            "allow_downsample": self.allow_downsample,
            "allow_upsample": self.allow_upsample,
            "uses_cmvn": self.uses_cmvn,
            "uses_deltas": self.uses_deltas,
            "uses_voiced": self.uses_voiced,
            "uses_splices": self.uses_splices,
            "uses_speaker_adaptation": self.uses_speaker_adaptation,
            "use_pitch": self.use_pitch,
            "min_f0": self.min_f0,
            "max_f0": self.max_f0,
            "delta_pitch": self.delta_pitch,
            "penalty_factor": self.penalty_factor,
        }
        if self.uses_splices:
            options.update(
                {
                    "splice_left_context": self.splice_left_context,
                    "splice_right_context": self.splice_right_context,
                }
            )
        return options

    # @abstractmethod
    # def calc_fmllr(self) -> None:
    #     """Abstract method for calculating fMLLR transforms"""
    #     ...

    @property
    def fmllr_options(self) -> MetaDict:
        """Options for use in calculating fMLLR transforms"""
        return {
            "fmllr_update_type": self.fmllr_update_type,
            "silence_weight": self.silence_weight,
            "silence_csl": getattr(
                self, "silence_csl", ""
            ),  # If we have silence phones from a dictionary, use them
        }

    @property
    def mfcc_options(self) -> MetaDict:
        """Parameters to use in computing MFCC features."""
        return {
            "use-energy": self.use_energy,
            "frame-shift": self.frame_shift,
            "frame-length": self.frame_length,
            "low-freq": self.low_frequency,
            "high-freq": self.high_frequency,
            "sample-frequency": self.sample_frequency,
            "allow-downsample": self.allow_downsample,
            "allow-upsample": self.allow_upsample,
            "snip-edges": self.snip_edges,
        }

    @property
    def pitch_options(self) -> MetaDict:
        """Parameters to use in computing MFCC features."""
        return {
            "use-pitch": self.use_pitch,
            "frame-shift": self.frame_shift,
            "frame-length": self.frame_length,
            "min-f0": self.min_f0,
            "max-f0": self.max_f0,
            "sample-frequency": self.sample_frequency,
            "penalty-factor": self.penalty_factor,
            "delta-pitch": self.delta_pitch,
            "snip-edges": self.snip_edges,
        }

