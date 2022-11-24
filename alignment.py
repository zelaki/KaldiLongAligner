from typing import List, Tuple, Optional
from kaldialign import align
from dataclasses import dataclass
from utils import IslandSegment, ctmEntry, labEntry

class T2TAlignment():

    def __init__(
        self,
        reference: List[str],
        hypothesis: List[str],
        hypothesis_ctm: List[ctmEntry], 
        current_alignment: Optional[List[labEntry]],
        text_onset_index: int,
        segment_onset_time: float,
        init: bool
    ):

        self.reference = reference
        self.hypothesis = hypothesis
        self.hypothesis_ctm = hypothesis_ctm
        self.current_alignment = current_alignment
        self.text_onset_index = text_onset_index
        self.segment_onset_time = segment_onset_time
        self.init = init
    
    def text_to_text_align(self, reference, hypothesis, EPS: str = "*") -> str:
        """
        Parameters
        ----------
        reference: List[str]
            List of refrence words
        hypothesis: List[str]
            List of hypothesis words

        Returns
        -------
        str
            String sequence of correct (C), insertions (I),
            deletions (D), substitution (S)
        """
        C = 'C'
        I = 'I'
        D = 'D'
        S = 'S'
        
        # returns a list of tuples
        t2t = align(reference, hypothesis, EPS)

        alignment = ''
        for ref_word, hyp_word in t2t:
            if ref_word == hyp_word:
                alignment+=C
            elif ref_word == EPS:
                alignment+=I
            elif hyp_word == EPS:
                alignment+=D
            elif ref_word != hyp_word:
                alignment+=S
        return alignment


    def text_to_text_islands(self, alignment: str, island_length: int = 3) -> List[Tuple[int, int]]:
        """
        Calculates the islands of confidence given refrence and hypothesis

        Parameters
        ----------
        alignment: str
            String sequence of C,I,D,S (output from text_to_text_align)
        island_length:str
            Minimum number of consecutive aligned words
        
        Returns
        -------
        List[IslandSegment]
            List of start and end insex of each island
        """
        islands = []

        if len(alignment) in [1,2,3]:
            island = 'C'
        else:
            island='C'*island_length

        strIndex=0
        list=['D','I','S']
        while strIndex<len(alignment):
            try:
                curr=strIndex+alignment[strIndex:].index(island)
            except ValueError:
                break
            strIndex=curr+min(alignment[curr:].index(i) if i in alignment[curr:] else len(alignment)-curr for i in list)
            island_segment = IslandSegment(
                onset_index=curr-alignment[:curr].count('I'),
                offset_index=strIndex-1-alignment[:strIndex-1].count('I'))
            islands.append(island_segment)
        return islands

    def initialize_alignment(self) -> List[labEntry]:

        return [labEntry(word=word, onset=-1, offset=-1) for word in self.reference]


    def frames_to_seconds(self, frames: int, frame_shift: float = 0.01) -> float:
        
        return round(frames * frame_shift, 5)


    def update_alignment(
        self,
        # current_alignment: List[labEntry],
        # hypothesis_ctm: List[ctmEntry],
        reference_islands: List[IslandSegment],
        hypothesis_islands: List[IslandSegment]
        # text_onset_index: int,
        # segment_onset_time: float
        # log_path
        ):

        try:
            assert len(reference_islands) == len(hypothesis_islands)
        except ArithmeticError:
            print('The number ref and hyp islands are not equal')


        for ref_island, hyp_island in zip(reference_islands, hypothesis_islands):

            try:
                assert ref_island.offset_index - ref_island.onset_index == hyp_island.offset_index - hyp_island.onset_index
            except AssertionError:
                print('The number of words in the islands are not equal')
            


            ref_island_onset = ref_island.onset_index
            ref_island_offset = ref_island.offset_index
            hyp_island_onset = hyp_island.onset_index
            while ref_island_onset <= ref_island_offset:
                ctm_entry = self.hypothesis_ctm[hyp_island_onset]
                lab_entry = self.current_alignment[self.text_onset_index+ref_island_onset]
                try:
                    assert lab_entry.word == ctm_entry.word
                except:
                    print('Words in correct segments are not matching')
                
                lab_entry.onset = self.segment_onset_time + self.frames_to_seconds(ctm_entry.onset)
                lab_entry.offset = lab_entry.onset + self.frames_to_seconds(ctm_entry.duration)

                ref_island_onset+=1
                hyp_island_onset+=1
        return self.current_alignment

    def run(self):
        reference_t2t = self.text_to_text_align(self.reference, self.hypothesis)
        hypothesis_t2t = self.text_to_text_align(self.hypothesis, self.reference)
        reference_islands = self.text_to_text_islands(reference_t2t)
        hypothesis_islands = self.text_to_text_islands(hypothesis_t2t)
        self.current_alignment = self.initialize_alignment()
        current_alignment = self.update_alignment(
            reference_islands=reference_islands,
            hypothesis_islands=hypothesis_islands)
        return current_alignment