import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
# from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer

# from peft import (
#     LoraConfig,
#     get_peft_model,
#     get_peft_model_state_dict,
#     prepare_model_for_int8_training,
#     set_peft_model_state_dict,
# )


@registry.register_model("mini_gpt4")
class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna0": "configs/models/minigpt4_vicuna0.yaml",
        # "pretrain_llama2": "configs/models/minigpt4_llama2.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        has_qformer=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        lora_r=0,
        lora_target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        self.has_qformer = has_qformer
        if self.has_qformer:
            print('Loading Q-Former')
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features
            )
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.load_from_pretrained(url_or_filename=q_former_model)

            if freeze_qformer:
                for name, param in self.Qformer.named_parameters():
                    param.requires_grad = False
                self.Qformer = self.Qformer.eval()
                self.Qformer.train = disabled_train
                self.query_tokens.requires_grad = False
                logging.info("freeze Qformer")

            img_f_dim = self.Qformer.config.hidden_size
            print('Loading Q-Former Done')
        else:
            img_f_dim = self.visual_encoder.num_features * 4
            print('Do not use Q-Former here.')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = "$$"

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            print("llama_model: ", llama_model)
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        if lora_r > 0:
            self.llama_model = prepare_model_for_int8_training(self.llama_model)
            loraconfig = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llama_model = get_peft_model(self.llama_model, loraconfig)

            # if ckpt_path:
            #     print('load the llm under lora')
            #     ckpt = torch.load(ckpt_path)
            #     set_peft_model_state_dict(self.llama_model,ckpt)
            self.llama_model.print_trainable_parameters()

        else:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image, early_exit_layer_idx=None):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            final_layer_features, early_exit_features = self.visual_encoder(
                image, early_exit_layer_idx
            )
            # early_exit_layers not activated
            if early_exit_features == None:
                image_embeds = self.ln_vision(final_layer_features).to(device)
            else:
                # print("early_exit_features", len(early_exit_features))
                # image_embeds = self.ln_vision(early_exit_features[early_exit_layer_idx]).to(device)
                image_embeds = self.ln_vision(early_exit_features[0]).to(device)

            # image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            if self.has_qformer:
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                inputs_llama = self.llama_proj(query_output.last_hidden_state)
            else:
                image_embeds = image_embeds[:, 1:, :]
                bs, pn, hs = image_embeds.shape
                image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))

                inputs_llama = self.llama_proj(image_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def get_context_emb(self, prompt, img_list):
        device = img_list[0].device
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)

        return mixed_embs

    def prompt_wrap(self, img_embeds, atts_img, prompts):
        if prompts:
            emb_lists = []
            if isinstance(prompts, str):
                prompts = [prompts] * len(img_embeds)

            for each_img_embed, each_prompt in zip(img_embeds, prompts):
                p_before, p_after = each_prompt.split('<ImageHere>')

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_before_embed = self.embed_tokens(p_before_tokens.input_ids)
                p_after_embed = self.embed_tokens(p_after_tokens.input_ids)
                wrapped_emb = torch.cat([p_before_embed, each_img_embed[None], p_after_embed], dim=1)
                emb_lists.append(wrapped_emb)
            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)
            if self.llama_tokenizer.padding_side == "right":
                for i, emb in enumerate(emb_lists):
                    wrapped_embs[i, :emb_lens[i]] = emb
                    wrapped_atts[i, :emb_lens[i]] = 1
            else:
                for i, emb in enumerate(emb_lists):
                    wrapped_embs[i, -emb_lens[i]:] = emb
                    wrapped_atts[i, -emb_lens[i]:] = 1
            return wrapped_embs, wrapped_atts, p_before_embed.shape[1]
        else:
            return img_embeds, atts_img, 1

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens

    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)

        if self.prompt_list:
            instruction = random.choice(self.prompt_list)
        else:
            instruction = samples["instruction_input"] if "instruction_input" in samples else None

        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, instruction)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["answer"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(img_embeds, atts_img, to_regress_embeds, to_regress_tokens.attention_mask)
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

        part_targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        targets = (
            torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                       dtype=torch.long).to(image.device).fill_(-100)
        )

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i] + 1:input_lens[i] + len(target) + 1] = target  # plus 1 for bos

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        max_new_tokens=300,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1,
        num_captions=1,
        temperature=1,
        past_key_values=None,
        output_attentions=True,
        output_hidden_states=True, 
        return_dict_in_generate=True,

        nvcd=True,  
        nvcd_previous_last_ids_list=[],

        key_position=None,

        # premature_layer=None,
        # candidate_premature_layers=None,
        # mature_layer=None,
        # beam_search=False,
        # dola_decoding = False,
        # halc_decoding = False,
        # opera_decoding=False,
        # vcd_decoding=False,
        # halc_assistant=None,
        
        # scale_factor=1.0,
        # threshold=1,
        # num_attn_candidates=5,
        # penalty_weights=1.0,
        # # VCD
        # images_cd=None,
        # cd_alpha=1,
        # cd_beta=0.1
    ):
        self.llama_tokenizer.padding_side = "left"
        self.model_name = "minigpt4"
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)

        if self.prompt_list:
            instruction = random.choice(self.prompt_list)
        else:
            instruction = samples["prompt"] if "prompt" in samples else None # e.g., prompt = ["<Img><ImageHere></Img> Is there a dog?", "<Img><ImageHere></Img> Is there a cat?", ...]

        self.instructions = instruction

        inputs_embeds, attention_mask, img_start_pos = self.prompt_wrap(img_embeds, atts_img, instruction)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=torch.int64,
                         device=inputs_embeds.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = attention_mask[:, :1]

        with self.maybe_autocast():

            ####여기다가 if nvcd: 면 input_ids에다가 이전 아웃풋의 마지막걸 추가해서 다음을 만듦.
            #nvcd가 True이고 이게 존재하면, 이걸 최초인풋에 붙여서 다음1토큰을 생성하는 모드임.
            if nvcd and len(nvcd_previous_last_ids_list) > 0:
                max_new_tokens = 1
                nvcd_previous_last_tokens = torch.tensor([nvcd_previous_last_ids_list], dtype=torch.int64, device=image.device)
                nvcd_previous_last_tokens_embeds = self.embed_tokens(nvcd_previous_last_tokens)
                inputs_embeds = torch.cat([bos_embeds, inputs_embeds, nvcd_previous_last_tokens_embeds], dim=1)
            else:
                inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)


            # attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

            if key_position is None:
                key_position = {
                    "image_start": img_start_pos+1,
                    "image_end": img_start_pos+img_embeds.shape[1],
                    "response_start": inputs_embeds.shape[1]
                }

            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                # minigpt는 input_ids가 아니라 llava와 같은 kv캐싱이 안됨..일단은 캐싱 없이 생성.
                # use_cache=True,
                # past_key_values=past_key_values,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                # max_length=max_length,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,

                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, 
                return_dict_in_generate=return_dict_in_generate,

                key_position=key_position,

                # premature_layer=premature_layer,
                # candidate_premature_layers=candidate_premature_layers,
                # mature_layer=mature_layer,
                # beam_search=beam_search,
                # dola_decoding=dola_decoding,
                # halc_decoding=halc_decoding,
                # opera_decoding=opera_decoding,
                # vcd_decoding=vcd_decoding,
                # halc_assistant=halc_assistant,
                # # opera
                # key_position=key_position,
                # scale_factor=scale_factor,
                # threshold=threshold,
                # num_attn_candidates=num_attn_candidates,
                # penalty_weights=penalty_weights,
                # # VCD
                # images_cd=images_cd,
                # cd_alpha=cd_alpha,
                # cd_beta=cd_beta,
                # LVLM_backbone=self,
            )
        # outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        # outputs[outputs == 1] = 2 # convert output id 1 to 2 (eos_token_id)
        # output_text = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # output_text = [text.split('###')[0].split('Assistant:')[-1].strip() for text in output_text]
        
        output_ids = outputs['sequences']
        output_attentions = outputs['attentions']
        output_hidden_states = outputs['hidden_states']
        output_past_key_values = outputs['past_key_values']

        output_ids[output_ids == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_ids[output_ids == 1] = 2 # convert output id 1 to 2 (eos_token_id)
        output_ids = output_ids.to(inputs_embeds.device)

        # input_token_len = inputs_embeds.shape[1]
        # print('inputs_embeds.shape모양!!!!!', inputs_embeds.shape)

        # reduced_inputs_embeds = inputs_embeds.mean(dim=-1)  # (1, 48, 4096) -> (1, 48)
        # n_diff_input_output = (reduced_inputs_embeds != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        ######
        vocab_size = self.llama_tokenizer.vocab_size

        all_output_ids = torch.where(output_ids < 0, output_ids + vocab_size, output_ids)
        # minigpt4는 입력, 출력을 다 출력하지않고 출력부만 뽑아주는듯? 
        # 일단 그럼 입력부랑 출력부 같게 설정함함
        input_token_ids = all_output_ids[:, :]
        output_token_ids = all_output_ids[:, :]

        return {"sequences" : all_output_ids,
                "attentions" : output_attentions,
                "hidden_states" : output_hidden_states,
                'past_key_values' : output_past_key_values,
                "input_token_ids" : input_token_ids,
                "output_token_ids" : output_token_ids,
                # 'before_image_token_len' : before_image_token_len,
                # 'image_token_len' : image_token_len,
                # 'after_image_token_len' : after_image_token_len
                }

    def embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        has_qformer = cfg.get("has_qformer", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 0)
        lora_alpha = cfg.get("lora_alpha", 32)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            has_qformer=has_qformer,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
