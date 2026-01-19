import difflib,torch
import numpy as np
from typing import Iterable, Dict, List, Optional, Tuple
from tqdm import tqdm
from allennlp.data import TokenIndexer, Tokenizer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField,ListField, TensorField,ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token

from utils.data_utils.line_align_utils import align_lines_in_diff_jointly, align_lines_in_diff_two_streams,align_lines_in_diff_jointly1
from utils.data_utils.op_mask_utils import diff_and_gen_op_mask
from utils.diff import remake_diff_from_hunks
from utils.file import read_dumped

def filter_empty_tokens_in_field(field):
    if isinstance(field, TextField):
        filtered_tokens = [t for t in field.tokens if  t.text != ""]
        return TextField(filtered_tokens, field._token_indexers)
    elif isinstance(field, ListField):
        filtered_subfields = [filter_empty_tokens_in_field(f) for f in field.field_list]
        return ListField(filtered_subfields)
    else:
        return field

@DatasetReader.register("imp_base")
class ImpBaseDatasetReader(DatasetReader):
    '''
    Implicit code change dataset reader for apca task.
    An instance must contain 'add', 'del' and 'label' fields.
    'Op' field is optional for aligned sequences, and it can
    be None for unaligned sequences.
    '''

    def __init__(
            self,
            code_indexer: TokenIndexer,
            code_tokenizer: Tokenizer,
            max_tokens: int = 128,  # only valid for 'flat' diff structure
            empty_code_token: Optional[str] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.code_tokenizer = code_tokenizer
        self.code_token_indexers = {"code_tokens": code_indexer}  # or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.empty_code_token = empty_code_token
        self.differ = difflib.Differ()


    def make_diff_field_from_hunks(self, diff_hunks):
        diff_str = remake_diff_from_hunks(diff_hunks, self.line_joiner)
        # Empty diff lines
        if diff_str == '':
            diff_tokens = self.get_token_list_for_empty_code()
        else:
            diff_tokens = self.code_tokenizer.tokenize(diff_str)

        return TextField(diff_tokens[:self.max_tokens], self.code_token_indexers)


    def get_token_list_for_empty_code(self):
        """
        Process empty code input, to avoid NaN out of Transformer's forward.
        """
        if self.empty_code_token is None:
            tokens = self.code_tokenizer.tokenize('')
            # If len=2, it is probably PretrainedTransformerTokenizer that outputs <s> and </s>.
            # We add at least one non-functional token to tokens to prevent unmatched shape
            # between token_ids and mask caused by transformer indexer
            if len(tokens) == 2:
                tokens.insert(1, Token(''))
            # ensure not empty returned token list
            elif len(tokens) == 0:
                tokens.append(Token(''))
        else:
            # There is a empty code token, try to tokenize it with tokenizer, to
            # add some special tokens at head and tail
            tokens = self.code_tokenizer.tokenize(self.empty_code_token)

        return tokens


    def extract_add_del_code_hierarchy(self, commit_diffs) -> Tuple[List, List]:

        commit_add_diff, commit_del_diff = [], []
        for diff in commit_diffs:
            add_diff, del_diff = [], []
            for hunk in diff:
                add_lines_in_hunk, del_lines_in_hunk = [], []
                # list of string lines
                add_lines = hunk.get('added_code')
                del_lines = hunk.get('removed_code')

                for add_line in add_lines:
                    # Filter emtpy line
                    if add_line == '':
                        continue
                    add_lines_in_hunk.append(add_line)
                for del_line in del_lines:
                    # Filter emtpy line
                    if del_line == '':
                        continue
                    del_lines_in_hunk.append(del_line)

                add_diff.append(add_lines_in_hunk)
                del_diff.append(del_lines_in_hunk)

            commit_add_diff.append(add_diff)
            commit_del_diff.append(del_diff)

        return commit_add_diff, commit_del_diff


    def text_to_instance(self, patch: Dict) -> Instance:
        raise NotImplementedError


    def _read(self, file_path: str) -> Iterable[Instance]:
        patches = read_dumped(file_path)

        for patch in tqdm(patches, total=len(patches)):
            yield self.text_to_instance(patch)

@DatasetReader.register("imp_flat_combined")
class ImpFlatCombinedAlignDatasetReader(ImpBaseDatasetReader):
    def __init__(
        self,
        char_indexer: TokenIndexer,
        char_tokenizer: Tokenizer,
        token_indexer: TokenIndexer,
        token_tokenizer: Tokenizer,
        char_max_tokens: int = 512,
        max_tokens: int = None,
        hunk_separator: Optional[str] = None,
        line_joiner: str = '',
        use_op_mask: bool = True,
        op_mask_attend_first_token: bool = True,
        **kwargs
    ):
        kwargs.pop("code_indexer", None)
        kwargs.pop("code_tokenizer", None)
        super().__init__(token_indexer, token_tokenizer, None, **kwargs)
        self.char_indexer = char_indexer
        self.char_tokenizer = char_tokenizer
        self.token_indexer = token_indexer
        self.token_tokenizer = token_tokenizer
        self.char_max_tokens = char_max_tokens
        self.max_tokens = max_tokens
        self.hunk_separator = hunk_separator
        self.line_joiner = line_joiner
        self.use_op_mask = use_op_mask
        self.op_mask_attend_first_token = op_mask_attend_first_token
        self.line_align_func = align_lines_in_diff_jointly1

    # character
    def _character_flatten_and_make_text_field(self, code) -> ListField:
        patch_fields: List[TextField] = []
        for hunk in code:
            if hunk is None:
                continue
            for patch in hunk:
                if not patch:
                    continue

                lines_str = self.line_joiner.join(patch)
                tokens = self.char_tokenizer.tokenize(lines_str)
                tokens = [t for t in tokens if t.text != ""]
                if len(tokens) == 0:
                    continue
                patch_tf = TextField(tokens, self.code_token_indexers)
                patch_fields.append(patch_tf)

        if len(patch_fields) == 0:
            empty_tokens = self.get_token_list_for_empty_code()
            patch_fields = [TextField(empty_tokens, self.code_token_indexers)]
        return ListField(patch_fields)

    def _character_flatten_files_and_make_list_field(self, file_diffs: List[List[List[List[str]]]]) -> ListField:
        file_fields: List[ListField] = []
        for file_diff in file_diffs:
            file_patch_list_field = self._character_flatten_and_make_text_field(file_diff)
            file_fields.append(file_patch_list_field)
        return ListField(file_fields)

    def extract_add_del_code_hierarchy1(self, commit_diffs, patch_width=15, patch_height=4) -> Tuple[List, List]:
        def split_code_2d_into_patches(code_lines: List[str], width: int, height: int) -> List[List[str]]:
            patches = []
            num_lines = len(code_lines)
            if num_lines == 0:
                return patches

            for h_start in range(0, num_lines, height):
                block_lines = code_lines[h_start:h_start + height]
                max_len = max(len(line) for line in block_lines) if block_lines else 0
                if max_len == 0:
                    continue
                for w_start in range(0, max_len, width):
                    patch = []
                    for line in block_lines:
                        if w_start < len(line):
                            sub_line = line[w_start:w_start + width]
                            patch.append(sub_line)
                    patches.append(patch)

            return patches

        commit_add_diff, commit_del_diff = [], []
        for diff in commit_diffs:
            add_diff, del_diff = [], []
            for hunk in diff:
                add_lines = hunk.get('added_code', [])
                del_lines = hunk.get('removed_code', [])
                add_patches = split_code_2d_into_patches(add_lines, patch_width, patch_height)
                del_patches = split_code_2d_into_patches(del_lines, patch_width, patch_height)

                if add_patches:
                    add_diff.append(add_patches)
                if del_patches:
                    del_diff.append(del_patches)
            commit_add_diff.append(add_diff)
            commit_del_diff.append(del_diff)
        return commit_add_diff, commit_del_diff

    def diff_and_gen_op_mask1(self,differ: difflib.Differ,
                             diff_add_field: ListField,
                             diff_del_field: ListField,
                             attend_first_token: bool = False):
        def build_patch_mask_for_file(file_add_field: ListField, file_del_field: ListField):
            add_text_list = [
                " ".join([t.text for t in field.tokens])
                for field in file_add_field.field_list
            ]
            del_text_list = [
                " ".join([t.text for t in field.tokens])
                for field in file_del_field.field_list
            ]
            add_set = set(add_text_list)
            del_set = set(del_text_list)

            patch_mask_list_add = []
            for add_patch in add_text_list:
                if add_patch in del_set:
                    patch_mask_list_add.append(0)
                    continue
                contained_by_del = any(
                    add_patch in del_patch
                    for del_patch in del_text_list
                )
                if contained_by_del:
                    patch_mask_list_add.append(0)
                    continue
                patch_mask_list_add.append(1)

            patch_mask_list_del = []
            for del_patch in del_text_list:
                if del_patch in add_set:
                    patch_mask_list_del.append(0)
                    continue
                contained_by_add = any(
                    del_patch in add_patch
                    for add_patch in add_text_list
                )
                if contained_by_add:
                    patch_mask_list_del.append(0)
                    continue
                patch_mask_list_del.append(1)
            return (
                ArrayField(np.array(patch_mask_list_add, dtype="int64")),
                ArrayField(np.array(patch_mask_list_del, dtype="int64")),
            )
        add_file_masks = []
        del_file_masks = []

        for file_add_field, file_del_field in zip(diff_add_field.field_list, diff_del_field.field_list):
            add_mask_field, del_mask_field = build_patch_mask_for_file(file_add_field, file_del_field)
            add_file_masks.append(add_mask_field)
            del_file_masks.append(del_mask_field)

        add_op_mask_field = ListField(add_file_masks)
        del_op_mask_field = ListField(del_file_masks)

        return add_op_mask_field, del_op_mask_field




    # token
    def _token_flatten_and_make_text_field(self, code) -> TextField:
        lines = []
        for hunk in code:
            for line in hunk:
                lines.append(line)
            if self.hunk_separator is not None:
                lines.append(self.hunk_separator)

        if len(lines) == 0:
            tokens = self.get_token_list_for_empty_code()
        else:
            lines_str = self.line_joiner.join(lines)
            tokens = self.token_tokenizer.tokenize(lines_str)

        return TextField(tokens[:self.max_tokens], self.code_token_indexers)
    def _token_flatten_files_and_make_list_field(self, file_diffs: List[List[List[str]]]) -> ListField:
        file_fields = []
        for file_diff in file_diffs:
            field = self._token_flatten_and_make_text_field(file_diff)
            file_fields.append(field)
        return ListField(file_fields)



    def text_to_instance(self, patch: Dict) -> Instance:
        diff = patch.get('diff')
        label = patch.get('label')

        add_diff, del_diff = self.extract_add_del_code_hierarchy(diff)
        diff_add_token_field = self._token_flatten_files_and_make_list_field(add_diff)
        diff_del_token_field = self._token_flatten_files_and_make_list_field(del_diff)
        

        aligned_diff = self.line_align_func(diff)
        add_diff1, del_diff1 = self.extract_add_del_code_hierarchy1(aligned_diff)


        diff_add_char_field = self._character_flatten_files_and_make_list_field(add_diff1)
        diff_del_char_field = self._character_flatten_files_and_make_list_field(del_diff1)


        diff_add_char_field = filter_empty_tokens_in_field(diff_add_char_field)
        diff_del_char_field = filter_empty_tokens_in_field(diff_del_char_field)

        

        fields = {
            'diff_add_char': diff_add_char_field,
            'diff_del_char': diff_del_char_field,
            'diff_add_token': diff_add_token_field,
            'diff_del_token': diff_del_token_field,
        }


        if label is not None:
            fields['label'] = LabelField(int(label), skip_indexing=True)

        if self.use_op_mask:
            add_token_op_mask_field, del_token_op_mask_field = diff_and_gen_op_mask(
                self.differ,
                diff_add_token_field,
                diff_del_token_field,
                self.op_mask_attend_first_token
            )
            fields['add_token_op_mask'] = add_token_op_mask_field
            fields['del_token_op_mask'] = del_token_op_mask_field

            add_char_op_mask_field, del_char_op_mask_field = self.diff_and_gen_op_mask1(self.differ,
                                                                        diff_add_char_field, diff_del_char_field,
                                                                        self.op_mask_attend_first_token)
            fields['add_char_op_mask'] = add_char_op_mask_field
            fields['del_char_op_mask'] = del_char_op_mask_field

        return Instance(fields)


    


        
    
      

        

    