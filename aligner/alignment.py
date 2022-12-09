from typing import List, Tuple, Optional
from kaldialign import align
from dataclasses import dataclass
from aligner.utils import IslandSegment, ctmEntry, labEntry, UnaliRegion

class T2TAlignment():



    
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
        # if alignment[0] == 'C':
        #     cur_island_len = 1
        #     on_island = True
        #     island_onset = 0
        # else:
        #     cur_island_len = 0
        #     on_island = False
        # islands = []
        # for idx, sym in enumerate(alignment[1:]):

        #     if sym == 'C' and not on_island:
        #         cur_island_len = 1
        #         island_onset = idx + 1
        #     elif sym == 'C' and on_island:
        #         cur_island_len += 1
        #     elif sym != 'C' and on_island and cur_island_len >= island_length:
        #         islands.append(
        #             IslandSegment(
        #                 onset_index=island_onset,
        #                 offset_index=idx
        #             )
        #         )

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
        ) -> List[labEntry]:
        """
        Updates the current alignment

        Parameters
        ----------
        current_alignment: List[labEntry]
            current alignment represented as a list of lab format entries
        hypothesis_ctm: List[ctmEntry]
            Time aligned hypothesis in ctm format
        reference_island:
            List of onsets and offsets of aligned regions with respect to reference
        hypothesis_islands:
            List of onsets and offsets of aligned regions with respect to hypothesis
        text_onset_index:
            Index of first word
        segment_onset_time:
            Timing of first word

        Returns
        -------
        List[labEntry]
            Current alignment updated
        """

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
                
                # lab_entry.onset = segment_onset_time + self.frames_to_seconds(ctm_entry.onset)
                # lab_entry.offset = lab_entry.onset + self.frames_to_seconds(ctm_entry.duration)

                lab_entry.onset = segment_onset_time + ctm_entry.onset
                lab_entry.offset = lab_entry.onset + ctm_entry.duration



                ref_island_onset+=1
                hyp_island_onset+=1
        return current_alignment

    def get_unaligned_regions(
        self,
        current_alignemnt: List[labEntry]
    ) -> List[UnaliRegion]:
        """
        Returns a list of unaligned regions
        !!!THIS IS A HORRIBLY WRITTEN FUNCTION!!!

        Parameters
        ----------
        current_alignment: List[labEntry]
            current alignment represented as a list of lab format entries
        
        Returns
        -------
        List[UnaliRegion]
            List of unaligned regions
        """

    

        unaligned_regions = []
        if current_alignemnt[0].onset == -1:
            unali_reg_onset = 0
            unali_reg_flag = True
        else:
            unali_reg_flag = False
        
        for idx, entry in enumerate(current_alignemnt[1:]):
            if entry.onset == -1 and not unali_reg_flag:
                unali_reg_onset = idx + 1
                unali_reg_flag = True
            elif entry.onset != -1 and unali_reg_flag:
                unaligned_regions.append(
                    UnaliRegion(
                        onset_index=unali_reg_onset,
                        offset_index=idx,
                        onset_time=.0 if unali_reg_onset==0
                            else current_alignemnt[unali_reg_onset-1].offset,
                        offset_time=entry.onset,
                    )
                )
                unali_reg_flag = False
        if current_alignemnt[-1].onset == -1:
            unaligned_regions.append(
                    UnaliRegion(
                        onset_index=unali_reg_onset,
                        offset_index=idx,
                        onset_time=current_alignemnt[unali_reg_onset-1].offset,
                        offset_time=-1,
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
            ) -> Tuple[List[labEntry], List[UnaliRegion]] :
        """
        Performs text to text alignment bettewn <reference> and <hypothesis>

        Parameters
        ----------
        current_alignment: List[labEntry]
            current alignment represented as a list of lab format entries
        hypothesis_ctm: List[ctmEntry]
            Time aligned hypothesis in ctm format
        reference_island:
            List of onsets and offsets of aligned regions with respect to reference
        hypothesis_islands:
            List of onsets and offsets of aligned regions with respect to hypothesis
        text_onset_index:
            Index of first word
        segment_onset_time:
            Timing of first word

        Returns
        -------
        Tuple[List[labEntry], List[UnaliRegion]]
            Current alignment updated and remaining unaligned regions
        """


        reference_t2t = self.text_to_text_align(reference, hypothesis)
        hypothesis_t2t = ''
        for sym in reference_t2t:
            if sym == 'D':
                hypothesis_t2t+='I'
            elif sym == 'I':
                hypothesis_t2t+='D'
            else:
                hypothesis_t2t+=sym
        # hypothesis_t2t = self.text_to_text_align(hypothesis, reference)

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
        unaligned_regions = self.get_unaligned_regions(current_alignment)
        
        return current_alignment, unaligned_regions


