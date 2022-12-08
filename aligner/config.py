
import os
from dataclasses import dataclass 
from typing import Dict, Any, Tuple

MetaDict = Dict[str, Any]


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
class FmllrDecodingArgs():
    word_insertion_penalty: float
    frame_shift: float
    max_active: int
    beam: float
    lattice_beam: float
    words_path: str
    acoustic_scale: float
    final_model: str
    final_mat: str
    hclg_path: str
    spk2utt: str
    feats: str
    cmvn_ark: str
    cmvn_scp: str
    left_context: int
    right_context: int
    word_boundary_int: str
    feature_string: str
    lat_path: str
    fmllr_update_type: str
    sil_phones: str
    silence_weight: float
    pre_trans_path: str
    tmp_lat: str
    temp_trans_path: str
    trans_path: str
    final_lat_path: str


def create_fmllr_args(
    working_dir,
    model_dir_path,
    config) -> FmllrDecodingArgs:

    left_context, right_context = get_splice_context(model_dir_path)
    final_mat = os.path.join(model_dir_path, 'final.mat')
    args = FmllrDecodingArgs(
        word_insertion_penalty = config['decoding_options']['word_insertion_penalty'],
        frame_shift = 0.01,
        acoustic_scale = config['decoding_options']['acoustic_scale'],
        max_active = config['decoding_options']['max_active'],
        beam = config['decoding_options']['beam'],
        lattice_beam = config['decoding_options']['lattice_beam'],
        words_path = os.path.join(model_dir_path, 'words.txt'),
        final_model = os.path.join(model_dir_path, 'final.mdl'),
        final_mat = final_mat,
        hclg_path = os.path.join(working_dir, "graph_dir", "HCLG.fst"),
        spk2utt = os.path.join(working_dir,"utt2spk"),
        feats = os.path.join(working_dir,"feats.scp"),
        cmvn_ark = os.path.join(working_dir, "cmvn.ark"),
        cmvn_scp = os.path.join(working_dir,"cmvn.scp"),
        left_context=left_context,
        right_context=right_context,
        word_boundary_int = os.path.join(model_dir_path, 'word_boundary.int'),
        feature_string = f"ark,s,cs:apply-cmvn --utt2spk=ark:{working_dir}/utt2spk \
            scp:{working_dir}/cmvn.scp scp:{working_dir}/feats.scp ark:-| \
            splice-feats --left-context={left_context} --right-context={right_context} ark:- ark:- |\
            transform-feats {final_mat} ark:- ark:- |",
        lat_path = os.path.join(working_dir,'lat.1'),
        fmllr_update_type = "full",
        sil_phones = "1:2:3:4:5:6:7:8:9:10",
        silence_weight = 0.01,
        pre_trans_path = "pre_trans.1",
        tmp_lat = os.path.join(working_dir,'tmp_lat.1'),
        temp_trans_path= os.path.join(working_dir, 'trans_tmp'),
        trans_path = os.path.join(working_dir,"trans.1"),
        final_lat_path = os.path.join(working_dir,"fmllr_lat.1")
    )
    return args


@dataclass
class MFCCArgs():
    wav_path: str
    feats_scp_path: str
    splice_opts: Tuple[int, int]
    mfcc_options: MetaDict
    pitch_options: MetaDict
    final_matrix: str
    log_path: str


def get_splice_context(model_dir_path):
    with open(os.path.join(model_dir_path, 'splice_opts'), 'r') as f:
        line = f.readline().split()
        left_context = line[0].split('=')[1]
        right_context = line[1].split('=')[1]
    return left_context, right_context



def create_mfcc_args(model_dir_path, working_dir_path):

    features_config = FeatureConfigMixin()
    with open(os.path.join(model_dir_path, 'splice_opts'), 'r') as f:
        line = f.readline().split()
        left_context = line[0].split('=')[1]
        right_context = line[1].split('=')[1]

    args = MFCCArgs(
        wav_path = os.path.join(working_dir_path, 'wav.scp'),
        feats_scp_path = os.path.join(working_dir_path, 'feats.scp'),
        splice_opts=(left_context, right_context),
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

