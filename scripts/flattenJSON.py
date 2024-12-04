import json

def flatten_json(nested_json, parent_key='', sep='_'):
    items = []
    for k, v in nested_json.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

input_json = {
    "_commit_hash": None,
    "architectures": [
      "MusicgenForConditionalGeneration"
    ],
    "audio_encoder": {
      "_name_or_path": "facebook/encodec_32khz",
      "add_cross_attention": False,
      "architectures": [
        "EncodecModel"
      ],
      "audio_channels": 1,
      "bad_words_ids": None,
      "begin_suppress_tokens": None,
      "bos_token_id": None,
      "chunk_length_s": None,
      "chunk_size_feed_forward": 0,
      "codebook_dim": 128,
      "codebook_size": 2048,
      "compress": 2,
      "cross_attention_hidden_size": None,
      "decoder_start_token_id": None,
      "dilation_growth_rate": 2,
      "diversity_penalty": 0.0,
      "do_sample": False,
      "early_stopping": False,
      "encoder_no_repeat_ngram_size": 0,
      "eos_token_id": None,
      "exponential_decay_length_penalty": None,
      "finetuning_task": None,
      "forced_bos_token_id": None,
      "forced_eos_token_id": None,
      "hidden_size": 128,
      "id2label": {
        "0": "LABEL_0",
        "1": "LABEL_1"
      },
      "is_decoder": False,
      "is_encoder_decoder": False,
      "kernel_size": 7,
      "label2id": {
        "LABEL_0": 0,
        "LABEL_1": 1
      },
      "last_kernel_size": 7,
      "length_penalty": 1.0,
      "max_length": 20,
      "min_length": 0,
      "model_type": "encodec",
      "no_repeat_ngram_size": 0,
      "norm_type": "weight_norm",
      "normalize": False,
      "num_beam_groups": 1,
      "num_beams": 1,
      "num_filters": 64,
      "num_lstm_layers": 2,
      "num_residual_layers": 1,
      "num_return_sequences": 1,
      "output_attentions": False,
      "output_hidden_states": False,
      "output_scores": False,
      "overlap": None,
      "pad_mode": "reflect",
      "pad_token_id": None,
      "prefix": None,
      "problem_type": None,
      "pruned_heads": {},
      "remove_invalid_values": False,
      "repetition_penalty": 1.0,
      "residual_kernel_size": 3,
      "return_dict": True,
      "return_dict_in_generate": False,
      "sampling_rate": 32000,
      "sep_token_id": None,
      "suppress_tokens": None,
      "target_bandwidths": [
        2.2
      ],
      "task_specific_params": None,
      "temperature": 1.0,
      "tf_legacy_loss": False,
      "tie_encoder_decoder": False,
      "tie_word_embeddings": True,
      "tokenizer_class": None,
      "top_k": 50,
      "top_p": 1.0,
      "torch_dtype": "float32",
      "torchscript": False,
      "transformers_version": "4.31.0.dev0",
      "trim_right_ratio": 1.0,
      "typical_p": 1.0,
      "upsampling_ratios": [
        8,
        5,
        4,
        4
      ],
      "use_bfloat16": False,
      "use_causal_conv": False,
      "use_conv_shortcut": False
    },
    "decoder": {
      "_name_or_path": "",
      "activation_dropout": 0.0,
      "activation_function": "gelu",
      "add_cross_attention": False,
      "architectures": None,
      "attention_dropout": 0.0,
      "bad_words_ids": None,
      "begin_suppress_tokens": None,
      "bos_token_id": 2048,
      "chunk_size_feed_forward": 0,
      "classifier_dropout": 0.0,
      "cross_attention_hidden_size": None,
      "decoder_start_token_id": None,
      "diversity_penalty": 0.0,
      "do_sample": False,
      "dropout": 0.1,
      "early_stopping": False,
      "encoder_no_repeat_ngram_size": 0,
      "eos_token_id": None,
      "exponential_decay_length_penalty": None,
      "ffn_dim": 6144,
      "finetuning_task": None,
      "forced_bos_token_id": None,
      "forced_eos_token_id": None,
      "hidden_size": 1536,
      "id2label": {
        "0": "LABEL_0",
        "1": "LABEL_1"
      },
      "initializer_factor": 0.02,
      "is_decoder": False,
      "is_encoder_decoder": False,
      "label2id": {
        "LABEL_0": 0,
        "LABEL_1": 1
      },
      "layerdrop": 0.0,
      "length_penalty": 1.0,
      "max_length": 20,
      "max_position_embeddings": 2048,
      "min_length": 0,
      "model_type": "musicgen_decoder",
      "no_repeat_ngram_size": 0,
      "num_attention_heads": 24,
      "num_beam_groups": 1,
      "num_beams": 1,
      "num_codebooks": 4,
      "num_hidden_layers": 48,
      "num_return_sequences": 1,
      "output_attentions": False,
      "output_hidden_states": False,
      "output_scores": False,
      "pad_token_id": 2048,
      "prefix": None,
      "problem_type": None,
      "pruned_heads": {},
      "remove_invalid_values": False,
      "repetition_penalty": 1.0,
      "return_dict": True,
      "return_dict_in_generate": False,
      "scale_embedding": False,
      "sep_token_id": None,
      "suppress_tokens": None,
      "task_specific_params": None,
      "temperature": 1.0,
      "tf_legacy_loss": False,
      "tie_encoder_decoder": False,
      "tie_word_embeddings": False,
      "tokenizer_class": None,
      "top_k": 50,
      "top_p": 1.0,
      "torch_dtype": None,
      "torchscript": False,
      "transformers_version": "4.31.0.dev0",
      "typical_p": 1.0,
      "use_bfloat16": False,
      "use_cache": True,
      "vocab_size": 2048
    },
    "is_encoder_decoder": True,
    "model_type": "musicgen",
    "text_encoder": {
      "_name_or_path": "t5-base",
      "add_cross_attention": False,
      "architectures": [
        "T5ForConditionalGeneration"
      ],
      "bad_words_ids": None,
      "begin_suppress_tokens": None,
      "bos_token_id": None,
      "chunk_size_feed_forward": 0,
      "cross_attention_hidden_size": None,
      "d_ff": 3072,
      "d_kv": 64,
      "d_model": 768,
      "decoder_start_token_id": 0,
      "dense_act_fn": "relu",
      "diversity_penalty": 0.0,
      "do_sample": False,
      "dropout_rate": 0.1,
      "early_stopping": False,
      "encoder_no_repeat_ngram_size": 0,
      "eos_token_id": 1,
      "exponential_decay_length_penalty": None,
      "feed_forward_proj": "relu",
      "finetuning_task": None,
      "forced_bos_token_id": None,
      "forced_eos_token_id": None,
      "id2label": {
        "0": "LABEL_0",
        "1": "LABEL_1"
      },
      "initializer_factor": 1.0,
      "is_decoder": False,
      "is_encoder_decoder": True,
      "is_gated_act": False,
      "label2id": {
        "LABEL_0": 0,
        "LABEL_1": 1
      },
      "layer_norm_epsilon": 1e-06,
      "length_penalty": 1.0,
      "max_length": 20,
      "min_length": 0,
      "model_type": "t5",
      "n_positions": 512,
      "no_repeat_ngram_size": 0,
      "num_beam_groups": 1,
      "num_beams": 1,
      "num_decoder_layers": 12,
      "num_heads": 12,
      "num_layers": 12,
      "num_return_sequences": 1,
      "output_attentions": False,
      "output_hidden_states": False,
      "output_past": True,
      "output_scores": False,
      "pad_token_id": 0,
      "prefix": None,
      "problem_type": None,
      "pruned_heads": {},
      "relative_attention_max_distance": 128,
      "relative_attention_num_buckets": 32,
      "remove_invalid_values": False,
      "repetition_penalty": 1.0,
      "return_dict": True,
      "return_dict_in_generate": False,
      "sep_token_id": None,
      "suppress_tokens": None,
      "task_specific_params": {
        "summarization": {
          "early_stopping": True,
          "length_penalty": 2.0,
          "max_length": 200,
          "min_length": 30,
          "no_repeat_ngram_size": 3,
          "num_beams": 4,
          "prefix": "summarize: "
        },
        "translation_en_to_de": {
          "early_stopping": True,
          "max_length": 300,
          "num_beams": 4,
          "prefix": "translate English to German: "
        },
        "translation_en_to_fr": {
          "early_stopping": True,
          "max_length": 300,
          "num_beams": 4,
          "prefix": "translate English to French: "
        },
        "translation_en_to_ro": {
          "early_stopping": True,
          "max_length": 300,
          "num_beams": 4,
          "prefix": "translate English to Romanian: "
        }
      },
      "temperature": 1.0,
      "tf_legacy_loss": False,
      "tie_encoder_decoder": False,
      "tie_word_embeddings": True,
      "tokenizer_class": None,
      "top_k": 50,
      "top_p": 1.0,
      "torch_dtype": None,
      "torchscript": False,
      "transformers_version": "4.31.0.dev0",
      "typical_p": 1.0,
      "use_bfloat16": False,
      "use_cache": True,
      "vocab_size": 32128
    },
    "torch_dtype": "float32",
    "transformers_version": None
  }

flattened_json = flatten_json(input_json)

output_file = "flattened_config.json"
with open(output_file, "w") as f:
    json.dump(flattened_json, f, indent=4)

print(f"Flattened JSON saved to {output_file}")
