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
from minigpt4.models.base_model import BaseModel

import torch
from PIL import Image
from transformers import TextStreamer
import sys

sys.path.append("mPLUG-Owl/mPLUG-Owl2")
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from mplug_owl2.model.modeling_llama2 import replace_llama_modality_adaptive

NUM_IMAGE_TOKENS = 64

@registry.register_model("mplug-owl2")
class MPLUGOWL2(BaseModel):
    """
    MPLUG-OWL2 model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_llama2": "configs/models/mplug_owl2_llama2.yaml",
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

        # AutoModelForCausalLM.register(MPLUGOwl2Config, MPLUGOwl2LlamaForCausalLM)

        replace_llama_modality_adaptive()

        print("llama_model", llama_model)
        model_name = get_model_name_from_path(llama_model)
        tokenizer, model, image_processor, context_len = load_pretrained_model(llama_model, None, model_name, load_8bit=False, load_4bit=False, device="cuda:0")

        self.llm_tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

        print('Loading MPLUG-OWL2 Done')


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
        output_attentions=True, 
        output_hidden_states=True, 
        return_dict_in_generate=True,
        past_key_values=None,
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
        # key_position=None,
        # scale_factor=1.0,
        # threshold=1,
        # num_attn_candidates=5,
        # penalty_weights=1.0,
        # # VCD
        # images_cd=None,
        # cd_alpha=1,
        # cd_beta=0.1
    ):
        # self.llm_tokenizer.padding_side = "left"
        self.model_name = "mplug-owl2"
        image = samples["image"]
        # img_embeds, atts_img = self.encode_img(image)

        # if self.prompt_list:
        #     instruction = random.choice(self.prompt_list)
        # else:
        #     instruction = samples["prompt"] if "prompt" in samples else None # e.g., prompt = ["<Img><ImageHere></Img> Is there a dog?", "<Img><ImageHere></Img> Is there a cat?", ...]
        
        # self.instructions = instruction

        # inputs_embeds, attention_mask, img_start_pos = self.prompt_wrap(img_embeds, atts_img, instruction)

        # batch_size = img_embeds.shape[0]
        # bos = torch.ones([batch_size, 1],
        #                  dtype=torch.int64,
        #                  device=inputs_embeds.device) * self.llama_tokenizer.bos_token_id
        # bos_embeds = self.embed_tokens(bos)
        # atts_bos = attention_mask[:, :1]

        # with self.maybe_autocast():
        #     inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        #     attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

        #     if key_position is None:
        #         key_position = {
        #             "image_start": img_start_pos+1, 
        #             "image_end": img_start_pos+img_embeds.shape[1], 
        #             "response_start": inputs_embeds.shape[1]
        #         }
        # print("image.size", image.size)
        
        # image = Image.open(samples["img_path"]).convert('RGB')
        # max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        # image = image.resize((max_edge, max_edge))

        # image_tensor = process_images([image], self.image_processor)
        # image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        # image_tensor = image.to(self.model.device, dtype=torch.float16)

        # inp = DEFAULT_IMAGE_TOKEN + query
        # conv.append_message(conv.roles[0], inp)
        # conv.append_message(conv.roles[1], None)
        prompt = samples["prompt"] 

        input_ids = tokenizer_image_token(prompt, self.llm_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = "</s>"
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.llm_tokenizer, input_ids)
        streamer = TextStreamer(self.llm_tokenizer, skip_prompt=True, skip_special_tokens=True)

        # with torch.inference_mode():
        #     outputs = self.model.generate(
        #         input_ids,
        #         images=image,
        #         do_sample=True,
        #         temperature=temperature,
        #         max_new_tokens=max_new_tokens,
        #         streamer=streamer,
        #         use_cache=True,
        #         stopping_criteria=[stopping_criteria])
        chunks_before, chunks_after = [], []
        for p in [prompt]:
            chunk_before, chunk_after = p.split('<|image|>')
            chunks_before.append(chunk_before)
            chunks_after.append(chunk_after)

        # print("prompt", prompt)

        # print("chunks_before", chunks_before)
        # print("chunks_after", chunks_after)

        tokens_before = self.llm_tokenizer(
            chunks_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False
        ).to(image.device).input_ids

        tokens_after = self.llm_tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False
        ).to(image.device).input_ids

        # print("tokens_before", tokens_before.shape)
        # print("tokens_after", (tokens_after.shape))
        # print("input_ids", input_ids.shape)
        # input()

        with torch.inference_mode():     
            if key_position is None:
                # key_position = {
                #     "image_start": tokens_before.shape[1]+1, 
                #     "image_end": input_ids.shape[1]-tokens_after.shape[1], 
                #     "response_start": input_ids.shape[1]+1,
                # }
                key_position = {
                    "image_start": tokens_before.shape[1]+1, 
                    "image_end": tokens_before.shape[1]+NUM_IMAGE_TOKENS, 
                    "response_start": input_ids.shape[1]+NUM_IMAGE_TOKENS-1,
                }

            # print("key_position", key_position)
            # input()

            ####여기다가 if nvcd: 면 input_ids에다가 이전 아웃풋의 마지막걸 추가해서 다음을 만듦.
            #nvcd가 True이고 이게 존재하면, 이걸 최초인풋에 붙여서 다음1토큰을 생성하는 모드임.
            if nvcd and len(nvcd_previous_last_ids_list) > 0:
                max_new_tokens = 1
                nvcd_previous_last_tokens = torch.tensor([nvcd_previous_last_ids_list], dtype=torch.int64, device=image.device)
                input_ids = torch.cat([input_ids, nvcd_previous_last_tokens], dim=1)
            else:
                input_ids = torch.cat([input_ids], dim=1)

            outputs = self.model.generate(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=past_key_values,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                # max_length=max_length,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                # streamer=streamer,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                images=image,

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
                # use_cache=True,
                # stopping_criteria=[stopping_criteria],
            )


        # outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        # outputs[outputs == 1] = 2 # convert output id 1 to 2 (eos_token_id)
        # output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # output_text = self.llm_tokenizer.decode(outputs[0, input_ids.shape[1]:]).strip()
        
        # print("output_text", output_text)
        # input()
        # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        # output_text = [text.split('###')[0].split('Assistant:')[-1].strip() for text in output_text]
        
        # output_text = [output_text]

        output_ids = outputs['sequences']
        output_attentions = outputs['attentions']
        output_hidden_states = outputs['hidden_states']
        output_past_key_values = outputs['past_key_values']

        output_ids = output_ids.to(input_ids.device)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')


        ######

        vocab_size = self.llm_tokenizer.vocab_size

        all_output_ids = torch.where(output_ids < 0, output_ids + vocab_size, output_ids)
        input_token_ids = all_output_ids[:, :input_token_len]
        output_token_ids = all_output_ids[:, input_token_len:]

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
