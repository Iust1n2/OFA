# OFA model Implementation

## Pretraining

We will use OFA base: 

 * 182M params
 * ResNet101 vision backbone
 * Hidden Size:  768
 * Intermediate Size: 3072
 * No. of Heads: 12
 * Enc. Layers: 6
 * Dec. Layers: 6

script for running pretraining: 
```
cd run_scripts/pretraining
bash pretrain_ofa_base.sh
```

## 26 Mar
## Step 1. Preliminary data download (!) 

From `pretrain_ofa_base.sh` created inside dataset dir 4 TSV files for the pretraining datasets and a subfolder for negative samples.
In addition, the folder `negative_sample contains` three files `all_captions.txt`, `object.txt` and `type2ans.json`. The data in these files are used as negative samples for the image-text matching (ITM) task.

! Note: accessing the dataset link is not working on ICI wifi because of possible filters for potentially harmful sites. On mobile hotspot the download works at Step 3.

## Step 2. Install additional fairseq module 

Inside OFA-copy cloned Fairseq directory:
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

## Step 3. Downloading the data

Downloaded the pretrain_data_examples zip unzipped in pretrain_data

- vision_language_examples.tsv (!MAIN SOURCE - first argument in pretrain script): Each line contains uniq-id, image (base64 string), caption, question, answer, ground-truth objects (objects appearing in the caption or question), dataset name (source of the data) and task type (caption, qa or visual gronunding). Prepared for the pretraining tasks of visual grounding, grounded captioning, image-text matching, image captioning and visual question answering.

- text_examples.tsv: Each line contains uniq-id and text. Prepared for the pretraining task of text infilling.

- image_examples.tsv: Each line contains uniq-id, image (base64 string, should be resized to 256*256 resolution) and image-code (generate the sparse codes for the central part of image through VQ-GAN). Prepared for the pretraining task of image infilling.

- detection_examples.tsv: Each line contains uniq-id, image (base64 string) and bounding box annotations (contains the top-left and bottom-right coordinates of the bounding box, object_id and object_name, seperated by commas). Prepared for the pretraining task of detection.

Each dataset contains 100 examples


## 27 Mar 

## Step 4. Pretraining process

from `pretrain_ofa_base.sh` the order of tasks is: 

1. setting up data directories with the examples from `pretrain_data_examples.zip`
2. selecting the working columns from the tsv files (tab separated files - similar to csv)
3. specifying the task: `tasks/pretrain_tasks/unify_task.py`
4. selecting the architecture from `OFA/models/ofa/ofa.py`- ofa_base
5. setting pretraining hyperparams
6. save path
7. running the pretraining script

## Step 5. Data preprocessing and preparation

`unify_dataset` in /pretrain_data is the script responsible for the data preparation for pretraining. It uses the modules from /data `data_utils.py, ofa_dataset.py` and from /utils `transforms.py, vision_helper.py`: 

- `get_whole_word_mask` of params `bpe, dictionary`

- `collate` is the function that merges `src_tokens of src_length`, `patch_images, patch_masks` into batch:
```
batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "code_masks": code_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
        "conf": conf
    }
```
- then, collate is called in another function `collater`: 
    ```
    """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        samples_v1 = []   # containing image-text pairs
        samples_v2 = []   # containing detection data, text data and image data
    ```
    
- class `UnifyDataset` of param `OFADataset` uses methods for processing different modalities:
    - for image-text pair:
          - random resize to `scales` which is the repatched image, max_size=672
          - center crop
          - random augment of params: `augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'])`
          - normalize of mean and std of `0.5`
      
    - for pure image:
          - same normalize of 0.5

    - for detection:
          - random horizontal flip
          - large scale jitter
          - and same normalize

    - for visual grounding:
          - random resize
          - object center crop
          - normalize

      - process functions for each of the above modalities:
            - (!) text and image are treated as one modality in `process_image_text_pair`:
                - it employs an if else to search if the task is captioning, qa, visual grounding and detection and for each one it encodes the information:
                    ```
                    if type == 'caption':
            tgt_caption = self.pre_caption(caption, self.max_tgt_length)
            pos_src_caption = self.pre_caption(caption, self.max_src_length)
            neg_src_caption = self.pre_caption(self.get_negative_caption(caption, gt_objects), self.max_src_length)
            src_item = self.encode_text(" what does the image describe?")
            tgt_item = self.encode_text(" {}".format(tgt_caption))
            pos_src_item = self.encode_text(' does the image describe " {} "?'.format(pos_src_caption))
            neg_src_item = self.encode_text(' does the image describe " {} "?'.format(neg_src_caption))
                    ```
            - text only `process_pure_text`
            - image only `process_pure_image`
            - detection `process_detection`
            - (!) each returns the following format:
                    ```
                    example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
            }
                    ```
        - methods text data:
              - `word_starts`
              - `add_whole_word_mask`
        - noise adding `add_insertion noise`

       - finally, `collater`

```
class UnifyDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_tgt_length=30,
        seed=7,
        code_dict_size=8192,
        num_bins=1000,
        patch_image_size=384,
        code_image_size=128,
        pure_text_dataset=None,
        pure_image_dataset=None,
        detection_dataset=None,
        all_object_list=None,
        all_caption_list=None,
        type2ans_dict=None,
        ans2type_dict=None,
        max_image_size=512,
        mask_ratio=0.3,
        random_ratio=0.0,
        keep_ratio=0.0,
        mask_length="span-poisson",
        poisson_lambda=3.0,
        replace_length=1
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
```


`OFADataset` is a class from `ofa_dataset.py` built on top of `FairseqDataset` class and includes methods for:
    - length function
    - `encode_text` 
    - `pre_question` (preprocess): 
        - lowercases, removes punctuation ",.!?*#:;~", replaces - and / with spaces
        - removes extra spaces
        - removes trailing newline characters
        - truncates the question to a maximum number of words if specified.
    - `pre_caption` employs the same as for question
    


Inspected the datasets in a notebook on vscode - `about_data`

`unify_task` uses `unify_dataset` for representing the images and text in a common space and submodules from tasks/ofa_task.py for adapting to the ofa model: `ofa_task, ofa_config`


## 28 Mar

`unify_task.py` prepares the dataset wrapped in a `@dataclass` decorator : `UnifyConfig` for the OFA model: 
```
@dataclass
class UnifyConfig(OFAConfig)
```

`OFAConfig` is a class that uses as param `FairseqDataClass` that in short, provides a metadata configuration for the data: 

```
class OFAConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated path to data list, will be iterated upon during epochs "
                    "in round-robin manner; valid data are always in the last"
        },
    )
    selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "selected cols"},
    )
    bpe: Optional[str] = field(
        default='gpt2',
        metadata={"help": "which bpe to use"},
    )
    bpe_dir: Optional[str] = field(
        default=None,
        metadata={"help": "bpe dir"},
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    max_src_length: int = field(
        default=128, metadata={"help": "the maximum src sequence length"}
    )
    max_tgt_length: int = field(
        default=30, metadata={"help": "the maximum target sequence length"}
    )

    code_dict_size: int = field(
        default=8192, metadata={"help": "code dict size"}
    )
    patch_image_size: int = field(
        default=480, metadata={"help": "patch image size"}
    )
    orig_patch_image_size: int = field(
        default=256, metadata={"help": "patch image size"}
    )
    num_bins: int = field(
        default=1000, metadata={"help": "number of quantization bins"}
    )

    imagenet_default_mean_and_std: bool = field(
        default=False,
        metadata={"help": "imagenet normalize"},
    )
    constraint_range: Optional[str] = field(
        default=None,
        metadata={"help": "constraint range"}
    )
```

Also, `unify_task` uses another decorator `@register_task` that takes as args `@register_task("unify_task", dataclass=UnifyConfig)`. 
The `UnifyTask` class inherits from submodule `OFATask`. 

`OFATask` class in `ofa_task.py` inherits from `FairseqTask` and defines methods for: 
    - `setup_task`
    - `get_batch_iterator`
    - `build_model`: 
        - uses the BPE either from Bert or GPT2: 
            ```
            if self.cfg.bpe == 'bert':
            bpe_dict = {
                "_name": "bert",
                "bpe_vocab_file": os.path.join(self.cfg.bpe_dir, "vocab.txt"),
                "bpe_cased": False
            }
            bpe_dict = DictConfig(bpe_dict)
            self.bpe = self.build_bpe(bpe_dict)
        else:
            bpe_dict = {
                "_name": "gpt2",
                "gpt2_encoder_json": os.path.join(self.cfg.bpe_dir, "encoder.json"),
                "gpt2_vocab_bpe": os.path.join(self.cfg.bpe_dir, "vocab.bpe")
            }
            bpe_dict = DictConfig(bpe_dict)
            self.bpe = self.build_bpe(bpe_dict)
        return model
            ```
    - `build_generator` (? for decoder): 
        - defines searching (defaults to beam search) and sampling methods
    - `train_step`: performs forward and backward pass: 
        ```
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample, update_num=update_num)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
        ```
        


The `load_dataset` method in `UnifyTask` class  inherits the dataset params from `UnifyDataset` module.
then it strips the white spaces for objects.txt and all_captions.txt from dataset/negative_samples/
and defines a method `get_batch_iterator` for creating mini-batches with given size constraints from the dataset. The iterator `epoch_iter` also uses the dataset collater method `collate_fn` inherited from the class `EpochBatchIterator` from fairseq/data/iterators.py ln 230. 



# Step 6. OFA architecture 

architecture from `OFA/models/ofa/ofa.py`- ofa_base

The backbone architecture of the ofa model is the decorater: 
`@register_model("ofa")
class OFAModel(TransformerModel)`
It inherits the `TransformerModel` class from `.unify_transformer.py` in ln 143:
```
@register_model("unify_transformer")
class TransformerModel(FairseqEncoderDecoderModel)
    """
    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder
        (both from fairseq.models.transformer: transformer_encoder.py & transformer_decoder.py
```
`FairseqEncoderDecoder` model is the actual backbone 

## Positional Embedding

For positional embeddings is used the class `SinusoidalPositionalEmbedding` with args (embedding_dim, padding_idx, init_size = 1024) and ignores padding symbols, from fairseq/modules/sinusoidal_positional_embeddings.py. `get_embedding` method uses several key aspects: 
- halving the embedding because summing up each sine and cos embedding (chatgpt explanation: "The embedding dimension is halved (half_dim) because each position's embedding will consist of a pair of sine and cosine values, thus effectively doubling the embedding size when concatenated")
- scaling factor of `math.log(10000) / (half_dim - 1)` that determines how quickly the wavelengths of the sine and cosine functions increase. A logarithmic scale is used to give the model more fine-grained position information for closer tokens while still providing position information for tokens further apart ! key difference from the original Transformer PE
- `emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)` calculates the rate at which each sine and cosine wave progresses, giving each dimension of the half embedding its unique frequency
- the positions `torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1)` are multiplied by these rates `emb.unsqueeze(0)` to generate the raw values for the sine and cosine functions. Applying `torch.sin` and `torch.cos` to these values generates the final sinusoidal patterns.
- The sine and cosine embeddings are concatenated to form a complete positional embedding for each position. If the embedding dimension is odd, an additional zero-padding step ensures the final embeddings have the correct size
- If a `padding_idx` is specified, that position's embedding is set to zero, ensuring that padding tokens do not contribute to the model's computations.

! the `encoder_embed_dim` for base is 768 & `max_src_length` = 80


## 29 Mar

completion of steps for data and multi modality handling and unification

