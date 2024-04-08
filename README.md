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

# Step 5. Data preprocessing and preparation

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
                - it employs an if else to search if the task is `captioning`, `qa`, `visual grounding` and for each one it encodes the information:
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
        - methods for text data:
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


- `OFADataset` is a class from `ofa_dataset.py` built on top of `FairseqDataset` class and includes methods for:

    - length function
    - `encode_text` 
    - `pre_question` (preprocess): 
        * lowercases, removes punctuation ",.!?*#:;~", replaces - and / with spaces
        * removes extra spaces
        * removes trailing newline characters
        * truncates the question to a maximum number of words if specified.
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

The backbone architecture of the ofa model is the OFA Model class decorator: 
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

## Sinusoidal Positional Embedding

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

## 3 Apr

Running inference with OFA Base with HuggingFace Transformers in this [link](https://colab.research.google.com/drive/1Ho81RBV8jysZ7e0FhsSCk_v938QeDuy3?usp=sharing). Did not work error `unauthorized access for huggingface login` on colab.

## OFAModel class - ofa.py

in `ofa.py` in ln 26 `OFAModel(TransformerModel)` class is initialized with BERT's random weights - from `fairseq/transformer_sentence_encoder.py`:

```
def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)
```

then checks if encoder has a dictionary attr and sets the `eos` to the one in the dictionary of the encoder.
then adds arguments in the `add_args` fn.
then defines the `forward` method which uses the encoder and decoder outputs:
```
encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            patch_images=patch_images,
            patch_masks=patch_masks,
            patch_images_2=patch_images_2,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens,
            sample_patch_num=sample_patch_num
        )
        x, extra = self.decoder(
            prev_output_tokens,
            code_masks=code_masks,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
```
continuation of the forward pass from ln 110 to 123: 

The line `pad = self.encoder.padding_idx` is getting the padding index from the encoder and storing it in the variable pad. Padding is a common technique in natural language processing where you add special tokens to the input to make all inputs the same length for batch processing.

The if `classification_head_name is not None`: line is checking if a classification head name has been provided. A classification head is a part of a model that is responsible for making the final prediction. It's usually a fully connected layer.

The `prev_lengths = prev_output_tokens.ne(pad).sum(1)` line is calculating the lengths of the previous output tokens by counting the number of tokens that are not equal to the padding index. The `.ne(pad)` part is creating a boolean mask where True indicates the token is not a padding token, and `.sum(1)` is summing this mask along the dimension 1, effectively counting the number of True values.

The `gather_index = prev_lengths[:, None, None].expand(x.size(0), 1, x.size(2)) - 1` line is creating an index for gathering values from the tensor x. It's expanding the prev_lengths tensor to match the size of x and then subtracting 1.

The `sentence_representation = x.gather(1, gather_index).squeeze()` line is using the gather_index to select specific values from the tensor x. The `.squeeze()` part is removing dimensions of size 1 from the shape of the tensor.

The `if self.classification_heads[classification_head_name].use_two_images`: line is checking if the classification head specified by classification_head_name uses two images. If it does, the following lines of code reshape the sentence_representation.

The for `k, head in self.classification_heads.items():` line is iterating over the items in self.classification_heads, which is a ModuleDict. A ModuleDict is a dictionary-like container for PyTorch modules. The items() method returns an iterable of the ModuleDict's key/value pairs.

The if `k == classification_head_name`: line is checking if the current key k is equal to classification_head_name. If it is, the following line `x = head(sentence_representation)` applies the corresponding classification head to the sentence_representation, and the loop is broken with break.

then, fn `register_embedding_tokens` initializes an empty answers tensor list and then iterates over the `ans2label_dict` and encodes an answer into a tensor and append it to the ans_tensor_list. it first retrieves an answer from the source dict, applies some preprocessing - it's slicing the string to remove the first 5 characters and the last character, and then replacing all underscores with spaces. then, it is encoding the preprocessed answer with the selected BPE into a tensor using the `encode_line` method of the src_dict. The line argument is the answer string that has been lowercased and prefixed with a space. finally, the encoded answer is appended to the `ans_tensor_list`.

then, fn `register_classification_head` is registering depending on the task a `mlp` or `linear` head for sentence-level classification tasks - in lines 270: 
```
if pooler_classifier == "mlp":
    self.dense = nn.Linear(input_dim, inner_dim)
    self.activation_fn = utils.get_activation_fn(activation_fn)
    self.dropout = nn.Dropout(p=pooler_dropout)
    self.out_proj = nn.Linear(inner_dim, num_classes)
elif pooler_classifier == "linear":
    self.dropout = nn.Dropout(p=pooler_dropout)
    self.out_proj = nn.Linear(input_dim, num_classes)
```

and each classification head has these attributes:
```
self.classification_heads[name] = OFAClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            pooler_classifier=self.args.pooler_classifier,
            use_two_images=use_two_images,
            do_spectral_norm=getattr(
                self.args, "spectral_norm_classification_head", False
```


then functions for managing the state_dict for registering token embeddings and new classification heads.

then in lines  322-380 the args for each OFA variant are declared, from which OFA Large stands as the baseline, containing args for encoder and decoder layers, activations, resnet vision layers and scales, and each variant including ours - OFABase inherits most of them:
```
@register_model_architecture("ofa", "ofa_large")
def ofa_large_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.0)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.pooler_classifier = getattr(args, "pooler_classifier", "mlp")

    args.resnet_drop_path_rate = getattr(args, "resnet_drop_path_rate", 0.0)
    args.encoder_drop_path_rate = getattr(args, "encoder_drop_path_rate", 0.0)
    args.decoder_drop_path_rate = getattr(args, "decoder_drop_path_rate", 0.0)

    args.resnet_type = getattr(args, "resnet_type", "resnet152")
    args.token_bucket_size = getattr(args, "token_bucket_size", 256)
    args.image_bucket_size = getattr(args, "image_bucket_size", 42)

    args.freeze_encoder_embedding = getattr(args, "freeze_encoder_embedding", False)
    args.freeze_decoder_embedding = getattr(args, "freeze_decoder_embedding", False)
    args.add_type_embedding = getattr(args, "add_type_embedding", True)
    args.attn_scale_factor = getattr(args, "attn_scale_factor", 2)

    args.code_image_size = getattr(args, "code_image_size", 128)
    args.patch_layernorm_embedding = getattr(args, "patch_layernorm_embedding", True)
    args.code_layernorm_embedding = getattr(args, "code_layernorm_embedding", True)
    args.entangle_position_embedding = getattr(args, "entangle_position_embedding", False)
    args.disable_entangle = getattr(args, "disable_entangle", False)
    args.sync_bn = getattr(args, "sync_bn", False)

    args.scale_attn = getattr(args, "scale_attn", False)
    args.scale_fc = getattr(args, "scale_fc", False)
    args.scale_heads = getattr(args, "scale_heads", False)
    args.scale_resids = getattr(args, "scale_resids", False)

    args.orig_patch_image_size = getattr(args, "orig_patch_image_size", 256)


@register_model_architecture("ofa", "ofa_base")
def ofa_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.resnet_type = getattr(args, "resnet_type", "resnet101")
    ofa_large_architecture(args)
```

### Unify Transformer - unify_transformer.py

UnifyTransformer is the builder for the unified model. Uses for layers of Encoder and Decoder fairseq classes: `FairseqEncoderDecoder` `FairseqEncoder` and `FairseqIncrementalDecoder`, which only defines the output of the Encoder and some modifiable methods for the forward pass and state dict update and beam search size. `FairseqIncrementalDecoder` is a special type of Decoder 

#### Encoder

`TransformerEncoder(FairseqEncoder)`:

Verifies if there is an encoder prompt and if not defines some attributes for `PromptEncoder` and attributes for the Encoder layers: `embed_dim, padding_idx, max_source_positions, num_attention_heads` then for attributes for text and image (for which we use `resnet101` in ln 604) embeddings:
```
elif args.resnet_type == 'resnet101':
    self.embed_images = ResNet(
        [3, 4, 23],
        norm_layer=norm_layer,
        drop_path_rate=args.resnet_drop_path_rate
    )
```

Then creates two embedding layers for images and text in lines 634-644: 
```
self.embed_positions = Embedding(
            args.max_source_positions + 2, embed_dim)
self.embed_image_positions = Embedding(
    args.image_bucket_size ** 2 + 1, embed_dim)
``` 

`Embedding` (line 1862) takes as args: `num_embeddings, embedding_dim, padding_idx, zero_init` with normal initialization: `nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)`

`build_encoder_layer` defines the Encoder layer inherited from `TransformerEncoderLayer` class in `unify_transformer_layer.py`:

`TransformerEncoderLayer` - Encoder layer block:

- uses `AdapterLayer` which is a class that implements bert weight initialization and multi-head attention head projections and relu non-linear function
- multi-head and FFN postprcoessing with dropout, layernorm and residual connections (! check docstring in ln 111)
- self-attention method for multi-head attentions by returning the class `MultiheadAttention` from module `unify_multihead_attention.py`:
    
#### Multi-Head Attention

class in module `unify_multihead_attention.py`

first initializes with embedding dimension, keys and values dims and query dim the same as key and values dim (! Self-attention requires query, key and " "value to be of the same size), then number of heads and each head dim as `embed_dim // num_heads` and head scaling.

after, k, v, q are projected `quant_noise` is applied - from `quant_noise.py`. Basically, quantization noise is applied to the weights as a pre-adaptation for the quantization in training: 

- Quantization Noise Parameter (p): This parameter determines the amount of quantization noise to apply. A value of p indicates the proportion of the weights within a block that will be set to zero (simulating the dropout of those weights due to quantization).

- Block Size (block_size): The function applies noise in blocks of weights, where the block size is determined by this parameter. The idea is that in practical quantization schemes (like Iterative Product Quantization, or iPQ), weights are quantized in blocks rather than individually.

- Noise Application Process:

    - The weights are divided into blocks according to the specified block_size.
    - A mask is created where each block has a probability p of being zeroed out.
    - The mask is applied to the weights, effectively dropping a portion of them to simulate quantization noise.
    - The weights are scaled by 1 / (1 - p) to maintain the original scale of the weight distribution, compensating for the reduction in effective weight magnitude due to the applied noise.

- Training-Only Application: The noise is applied only during the training phase (if mod.training). This makes sense because the goal is to prepare the model for quantization during inference. There is no need to apply quantization noise during evaluation or inference.

- Forward Pre-Hook: The function utilizes register_forward_pre_hook to apply the quantization noise right before each forward pass. This ensures that the model sees a slightly different version of its weights on each iteration, mimicking the instability introduced by quantization.

The `forward` method takes as input key, value and queries and outputs:
```
Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
```
```
tgt_len, bsz, embed_dim = query.size()
src_len = tgt_len
[...]
if key is not None:
    src_len, key_bsz, _ = key.size()
```

Then employs projections for q, k, v.

Lines 313-407: 

Attention weights are employed with the dot product between the query and key transposed - batch matrix multiplication: `attn_weights = torch.bmm(q, k.transpose(1, 2))` and a sparse mask is applied `attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, k.size(1), bsz)`. The weights are then placed in a list `assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, k.size(1)]` and then are applied some masks to the weights (in lines 348-378) before softmax: 

- Attention Mask (attn_mask): An optional mask that can additively modify the attention scores. It's used to prevent attention to certain positions.
- Self Attention Mask (self_attn_mask): This mask operates similarly to attn_mask, focusing on self-attention scenarios.
- Key Padding Mask: This is crucial for ignoring padding tokens in the input sequence. The mask is applied by setting the attention scores to -infinity for positions corresponding to padding tokens, effectively zeroing out these positions after softmax.

```
attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
attn_probs = self.dropout_module(attn_weights)
```

The attention probabilities are then used to compute a weighted sum of the value (v) vectors, resulting in the final attention output (attn) in `attn = torch.bmm(attn_probs, v)`. This represents the aggregated information from the input sequence as informed by the attention scores. and then listed in `assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]`. 

The aggregated attention output is then passed through an output projection `attn = self.out_proj(attn)`, aligning the dimensions with the expected output size and performing a linear transformation.

```
if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )
        if prompt_kv is not None:
            prompt_k, prompt_v = prompt_kv.split(1)
            prompt_k = prompt_k.squeeze(0).reshape(k.size(0), -1, k.size(2))
            prompt_v = prompt_v.squeeze(0).reshape(v.size(0), -1, v.size(2))
            k = torch.cat([prompt_k, k], dim=1)
            v = torch.cat([prompt_v, v], dim=1)
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, k.size(1), bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, k.size(1)]

        if attn_bias is not None:
            attn_weights[:, :, -src_len:] += attn_bias[:, :, -src_len:]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if self_attn_mask is not None:
            self_attn_mask = self_attn_mask.unsqueeze(1).expand(bsz, self.num_heads, tgt_len, k.size(1))
            attn_weights += self_attn_mask.contiguous().view(bsz * self.num_heads, tgt_len, k.size(1))

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, k.size(1))
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, k.size(1))

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        if self.c_attn is not None:
            attn = attn.view(tgt_len, bsz, self.num_heads, self.head_dim)
            attn = torch.einsum('tbhd,h->tbhd', attn, self.c_attn)
            attn = attn.reshape(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, k.size(1)
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights
```

Back to `TransformerEncoderLayer` after `build_self_attention` that returns MultiheadAttention illustrated before, there are employed methods for residual connections, state dict update and forward pass - lines 222-292: 

In `unify_transformer.py` after method `build_encoder_layer` which consists of the `TransformerEncoderLayer` class I just presented, the code proceeds to methods for getting the relative positional bias for text and image in lines 722-744. 

#### Positional Embedding for Text and Images

Then, `get_patch_images_info` fn employs the following: 

1. Embeds image patches `image_embed = self.embed_images(patch_images)` and extracts the height and width with the ResNet vision module: `elif args.resnet_type == 'resnet101':
            self.embed_images = ResNet(
                [3, 4, 23],
                norm_layer=norm_layer,
                drop_path_rate=args.resnet_drop_path_rate
            )`

2. Generating patch positions: 
    - Calculates the number of patches `image_num_patches` based on the height (h) and width (w) of the embedded image
    - A padding mask `image_padding_mask` is created for the patches, initialized to zeros (and converted to boolean values), indicating no padding initially
    - Positional indices `image_position_idx` for each patch are generated based on their order in the image grid and possibly adjusted by `self.args.image_bucket_size`, indicating a scaled positional encoding. The + 1 suggests that positions are 1-indexed. The `image_position_ids` is resulted by expanding `image_position_idx` by added an extra dimension - the expand method is creating a new tensor where the image position indices are repeated for each item in the batch. The resulting tensor has shape `batch_size, image_num_patches`, and it contains the same image position indices for each item in the batch.

3. Flattening and transposing - The embedded patches are flattened and transposed to have a shape suitable for processing by subsequent layers `image_embed = image_embed.flatten(2).transpose(1, 2)`, typically making the sequence of patches the primary dimension

4. Sampling patches (Optional): 
    - If sample_patch_num is specified, a subset of patches is randomly sampled for each image. This involves:
        - Generating a list of indices (patch_orders) for the selected patches.
        - Using these indices to gather the selected embeddings, padding mask, and positional ids for further processing.
        - Adjusting image_num_patches to reflect the number of sampled patches.

5. Positional Embedding Interpolation (Optional):
    - If interpolate_position is enabled and the number of patches exceeds the original number of patches (orig_num_patches), positional embeddings are interpolated to match the new grid size This involves:
        - Generating a grid of original positional ids (old_image_position_ids).
        - Embedding these positions (old_image_pos_embed) and reshaping them to match the original image's height and width (orig_hw).
        - Using bilinear interpolation (F.interpolate) to resize the positional embeddings to match the new grid size.
        - Expanding the interpolated positional embeddings to match the batch size.

6. Regular Positional Embedding: 
    - If interpolation is not applied, positional embeddings are generated directly from the calculated positional indices (image_position_ids) using a model-defined embedding function (self.embed_image_positions)

7. Returns the embedded image patches `image_embed`, the number of patches `image_num_patches`, the padding mask `image_padding_mask`, the positional ids `image_position_ids`, and the positional embeddings `image_pos_embed`

Then, `get_encoder_prompt` fn in lines 796-808 is processing prompt tokens within an encoder. The function starts by encoding the given prompt_tokens using a dedicated encoder. The output, `past_key_values`, presumably contains key and value pairs used in attention mechanisms, encoded in its last dimension. The shape of the encoder prompt is extracted by batch size `bsz` and sequence length `seqlen` and then `past_key_values` is reshaped to distribute its dimensions into a more structured format that separates layers, attention heads, and features per head. Dropout is applied to the `reshaped past_key_values` tensor to prevent overfitting by randomly setting elements to zero during training with a certain probability. The tensor is permuted to organize it into a structure that is expected by the downstream attention mechanism. The permutation `[2, 0, 3, 1, 4]` rearranges the dimensions to place the encoder layers dimension first, followed by batch size, attention heads, sequence length, and features per head. Finally, the tensor is split into two parts using `.split(2)`. Since the encoder layers were multiplied by 2 earlier to account for keys and values, this split likely separates the keys and values for use in the attention calculations.

The `forward_embedding` fn is likely to be responsible for the unification of the embedding space of linguistic and visual tokens as described in Chapter 3.1 of the paper: 

For **textual token** embedding: 
- **Token Embedding:** first, if not provided, textual tokens `src_tokens` are first embedded using `self.embed_tokens`
- **Scaling:** the embeddings are then scaled by `self.embed_scale` to control the magnitude
- **Positional Embedding:** If `pos_embed` is provided and `self.entangle_position_embedding` is enabled, it adds positional embeddings to the token embeddings
- **Type Embedding:** A type embedding might be added to differentiate between different types of tokens (e.g., distinguishing between two types of textual tokens or between text and padding)
- **Normalization and Dropout**

For **image token** embedding:
- **Projection**: `image_embed` (and optionally image_embed_2 for a second set of images) is passed through `self.image_proj`, which likely projects raw image embeddings to a dimension compatible with textual token embeddings
- Scaling and Positional Embedding: Like textual tokens, image embeddings are scaled, and positional embeddings are added if provided and enabled.
- **Type Embedding for Images**: The function differentiates between different types of embeddings (e.g., distinguishing image embeddings from textual embeddings) by adding type embeddings. Notably, for the second set of images (image_embed_2), a distinct value (fill_value=2) is used, indicating a different type or source of images.
- **Concatenation**: The processed image embeddings are concatenated with the textual token embeddings, merging visual and textual information into a single sequence. This concatenated **sequence (x)** and **its embedding (embed)** are returned, (!) now unified in a single embedding space.

Next, the `forward` is employed which uses the `forward_scriptable` method in lines 872-1078:

Takes as input tokens in the source language and their lengths in `src_tokens` and `src_lengths`, token embeddings and `return_all_hidden_states` and returns a dictionary of the last encoder output shape of `(src_len, batch, embed_dim)` in `encoder_out`, the positions of the padding elements of shape `(batch, src_len)` in `encoder_padding_mask`, the encoder embedding lookup of shape  `(src_len, batch, embed_dim)` in `encoder_embedding` and all intermediate hidden states of shape `(src_len, batch, embed_dim)` in `encoder_states`.

`forward_scriptable` fn actually performs the forward pass:

- inputs and outputs the same as `forward` above
- uses methods `get_encoder_prompt` and `get_patch_images_info` for token embedding for text and images
- then applies positional encoding with `forward_embedding` method for unification in lines 990
- applies the positional encoding (lines 1000-1050) as described in the paper: 
    1. "For positional information, we use two absolute position embeddings for text and images, respectively ... In addition, we also use 1D relative position bias for text and 2D relative position bias for image"
    2. "Instead of simply adding the position embeddings, we decoupling the position correlation from token embeddings and patch embeddings"

i. code snippet
```
# lines 1011
## absolute position embedding 
pos_q = self.pos_q_linear(pos_embed).view(
    pos_embed.size(0), pos_embed.size(1), self.num_attention_heads, -1
).transpose(1, 2) * self.pos_scaling
pos_k = self.pos_k_linear(pos_embed).view(
    pos_embed.size(0), pos_embed.size(1), self.num_attention_heads, -1
).transpose(1, 2)
abs_pos_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))

# lines 1027
## first for text (src_tokens)
for idx, layer in enumerate(self.layers):
            self_attn_bias = abs_pos_bias.clone()
            self_attn_bias[:, :, -
                           src_tokens.size(1):, -
                           src_tokens.size(1):] += self.get_rel_pos_bias(src_tokens, idx)
            # then for image (image_position_ids)
            if patch_images_2 is not None:
                self_attn_bias[:, :, :image_num_patches_2, :image_num_patches_2] += \
                    self.get_image_rel_pos_bias(image_position_ids_2, idx)
                self_attn_bias[:, :, image_num_patches_2:image_num_patches_2 +
                               image_num_patches, image_num_patches_2:image_num_patches_2 +
                               image_num_patches] += self.get_image_rel_pos_bias(image_position_ids, idx)
            elif patch_images is not None:
                self_attn_bias[:, :, :x.size(0) - src_tokens.size(1), :x.size(
                    0) - src_tokens.size(1)] += self.get_image_rel_pos_bias(image_position_ids, idx)
            self_attn_bias = self_attn_bias.reshape(
                -1, self_attn_bias.size(2), self_attn_bias.size(2))
```
    
ii. code snippet
```
# line 640 
## first step - token embedding for text
self.embed_positions = Embedding(
            args.max_source_positions + 2, embed_dim)
# and for image
self.embed_image_positions = Embedding(
    args.image_bucket_size ** 2 + 1, embed_dim)

# line 867 decoupeling embeddings
x = torch.cat([image_x_2, x], dim=1)
embed = torch.cat([image_embed_2, embed], dim=1)
# line 990
## decoupeling positional embeddings
pos_embed = self.embed_positions(utils.new_arange(src_tokens))
pos_embed = torch.cat([image_pos_embed, pos_embed], dim=1)
# from which image_pos_embed is given from get_patch_image_info fn by extracting the positional information
```






