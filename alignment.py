from typing import List, Tuple, Optional
from kaldialign import align
from dataclasses import dataclass
from utils import IslandSegment, ctmEntry, labEntry, UnaliRegion

class T2TAlignment():

    # def __init__(
    #     self
    #     reference: List[str],
    #     # hypothesis: List[str],
    #     # hypothesis_ctm: List[ctmEntry], 
    #     # current_alignment: Optional[List[labEntry]],
    #     # text_onset_index: int,
    #     # segment_onset_time: float,
    #     # init: bool
    # ):

        # self.reference = reference
        # self.hypothesis = hypothesis
        # self.hypothesis_ctm = hypothesis_ctm
        # self.current_alignment = current_alignment
        # self.text_onset_index = text_onset_index
        # self.segment_onset_time = segment_onset_time
        # self.init = init

    
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

    def initialize_alignment(self, reference) -> List[labEntry]:

        return [labEntry(word=word, onset=-1, offset=-1) for word in reference]


    def frames_to_seconds(self, frames: int, frame_shift: float = 0.01) -> float:
        
        return round(frames * frame_shift, 5)


    def update_alignment(
        self,
        current_alignment: List[labEntry],
        hypothesis_ctm: List[ctmEntry],
        reference_islands: List[IslandSegment],
        hypothesis_islands: List[IslandSegment],
        text_onset_index: int,
        segment_onset_time: float
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
                ctm_entry = hypothesis_ctm[hyp_island_onset]
                lab_entry = current_alignment[text_onset_index+ref_island_onset]
                try:
                    assert lab_entry.word == ctm_entry.word
                except:
                    print('Words in correct segments are not matching')
                
                lab_entry.onset = segment_onset_time + self.frames_to_seconds(ctm_entry.onset)
                lab_entry.offset = lab_entry.onset + self.frames_to_seconds(ctm_entry.duration)

                ref_island_onset+=1
                hyp_island_onset+=1
        return current_alignment

    def get_unaligned_regions(
        self,
        reference_islands: List[IslandSegment],
        current_alignment: List[labEntry]
    ):

        unified_islands = []
        reference_islands_iter = iter(reference_islands[:-1])

        for idx, island in enumerate(reference_islands_iter):
            if island.offset_index == reference_islands[idx+1].onset_index - 1:
                unified_islands.append(
                    IslandSegment(
                        onset_index=island.onset_index,
                        offset_index=reference_islands[idx+1].offset_index
                    )
                )
                reference_islands_iter.__next__()
                idx+=idx
            else: 
                unified_islands.append(island)
        unified_islands.append(reference_islands[-1])

        unaligned_regions = []
        for island, nxt_island in zip(unified_islands, unified_islands[1:]):
            if island.onset_index != 0:
                unaligned_regions.append(
                    UnaliRegion(
                        onset_index=0,
                        offset_index=island.onset_index-1,
                        onset_time=.0,
                        offset_time=current_alignment[island.onset_index].onset
                    )
                )
            elif nxt_island.offset_index != len(current_alignment) - 1:
                unaligned_regions.append(
                    UnaliRegion(
                        onset_index=nxt_island.offset_index+1,
                        offset_index=len(current_alignment)-1,
                        onset_time=current_alignment[nxt_island.offset_index+1].offset,
                        offset_time=-1
                    )
                )
            unaligned_regions.append(
                UnaliRegion(
                    onset_index=island.offset_index+1,
                    offset_index=nxt_island.onset_index-1,
                    onset_time=current_alignment[island.offset_index].offset,
                    offset_time=current_alignment[nxt_island.onset_index].onset
                )
            )
        return unaligned_regions


        
    def run(self,
            reference: List[str],
            hypothesis: List[str],
            hypothesis_ctm: List[ctmEntry], 
            current_alignment: Optional[List[labEntry]],
            text_onset_index: int,
            segment_onset_time: float,
            ):
        reference_t2t = self.text_to_text_align(reference, hypothesis)
        hypothesis_t2t = self.text_to_text_align(hypothesis, reference)
        reference_islands = self.text_to_text_islands(reference_t2t)
        hypothesis_islands = self.text_to_text_islands(hypothesis_t2t)
        if current_alignment == None:
            current_alignment = self.initialize_alignment(reference=reference)

        current_alignment = self.update_alignment(
            current_alignment=current_alignment,
            hypothesis_ctm=hypothesis_ctm,
            reference_islands=reference_islands,
            hypothesis_islands=hypothesis_islands,
            text_onset_index=text_onset_index,
            segment_onset_time=segment_onset_time)

        unaligned_regions = self.get_unaligned_regions(reference_islands, current_alignment)
        
        return current_alignment, unaligned_regions


