
##### RVCD
git clone https://github.com/JiHoonLee9898/RVCD.git
cd RVCD
conda env create -f environment.yml
conda activate RVCD 
cd MAIN_CODES

##### LVLM backbones
https://huggingface.co/liuhaotian/llava-v1.5-7b 를, 
MAIN_CODES/eval_configs/prior_decoding_yamls/not_rvcd_llava.yaml,
MAIN_CODES/eval_configs/llava-1.5_eval.yaml의 14번째줄에 specify.

https://huggingface.co/Vision-CAIR/vicuna-7b 를,
MAIN_CODES/minigpt4/configs/models/minigpt4_vicuna0.yaml,
MAIN_CODES/minigpt4/configs/models/not_rvcd_minigpt4_vicuna0.yaml의 18번째줄에 specify.

https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b 를,
MAIN_CODES/eval_configs/mplug-owl2_eval.yaml,
MAIN_CODES/eval_configs/prior_decoding_yamls/not_rvcd_mplug_owl2.yaml의 14번째줄에 specify. 

RVCD/MAIN_CODES/prerained_minigpt4_7b.pth 를,
MAIN_CODES/eval_configs/minigpt4_eval.yaml,
MAIN_CODES/eval_configs/prior_decoding_yamls/not_rvcd_mini_gpt4_vicuna0.yaml의 8번째줄에 specify.

##### DINO for HALC
cd MAIN_CODES
export CUDA_HOME=$CONDA_PREFIX
cd decoder_zoo/GroundingDINO
pip install -e .
cd ../..
https://drive.google.com/drive/folders/1UaMJga-BKju88CXAdonbiQujBKkdcVGX 
다운받은 파일을
MAIN_CODES/decoder_zoo/GroundingDINO/weights/groundingdino_swint_ogc.pth
에 저장.

##### Arguments
RVCD/MAIN_CODES/run_example.sh의 구체적 예시를 참고. bash 파일 안의 각 블록들(총6개)은
각각 RVCD와 prior methods들의 CHIAR/BLEU,POPE,MME 평가를 위한 출력 캡션을 생성하게 함.

### RVCD Arguments detail
--model 
RVCD : llava-1.5, minigpt4, mplug-owl2
prior decoding methods : not_rvcd_llava, not_rvcd_mini_gpt4, not_rvcd_mplug_owl2

--ref_folder_path
절대경로 of 
RVCD/DB_single_concept_images_flux_generated/generated_images

--data_path
coco2014의 절대 경로. 

Note that [COCO_DIR] is expected to contain both images and annotation files within the annotations subfolder. In other words, [COCO_DIR] should the the following structure:

COCO_DIR (val2014 for example)
  - annotations
    - captions_val2014.json
    - captions_val2014.json
    - instances_train2014.json
    - instances_val2014.json
    - person_keypoints_train2014.json
    - person_keypoints_val2014.json
  - COCO_val2014_000000000042.jpg
  - COCO_val2014_000000000073.jpg
  ...

--coco_path
필요한 케이스(rvcd,mme)이면, 마찬가지로 coco2014의 절대 경로. 

--chair_cache_path
RVCD/MAIN_CODES/eval/CHAIR_CACHE/chair.pkl 의 절대 경로.

--yolo_version 
ultralytics의 'yolov8x.pt' 가 default detector.

--rvcd_alpha 1 Negative logits 제거 parameter
--rvcd_beta 0.1 Positive logits 회복 parameter

### Prior decoding methods Arguments detail
--model 
RVCD : llava-1.5, minigpt4, mplug-owl2
prior decoding methods : not_rvcd_llava, not_rvcd_mini_gpt4, not_rvcd_mplug_owl2

-d 
greedy, dola, halc, opera, vcd, beam

--data_paths 
data_path와 다름. data_paths는 MME벤치마크 데이터셋의 경로 제공

### halc additional arguments
--k-candidate-num	4	Number of generative focal fields for local search. Default: 4.
--expand-ratio	0.6	The growing factor of focal fields. Default: 0.6.
--detector	dino	Detector to use in [dino, owlv2]. Default: dino.
--box_threshold	0.4	The threshold for bounding box in GroundingDino. Default: 0.4.

### OPERA additional arguments
--scale_factor	50	The scale factor to scale up the self-attention weights. Default: 50.
--threshold	15	The threshold for attending retrospection. Default: 15.
--num_attn_candidates	5	The number of candidates per beam. Default: 5.
--penalty_weights	1	The weight of penalty term in decoding. Default: 1.

### VCD additional arguments
--cd-alpha	1	Amplification factor. Default: 1.
--cd-beta	0.1	Truncation factor for adaptive plausibility constraint. Default: 0.1.
--noise-step	500	Number of steps to add diffusion noise. Default: 500.

### EVALUATION
RVCD/MAIN_CODES/run_example.sh의 구체적 예시를 참고. bash 파일 안의 각 블록들(총6개)은
각각 RVCD와 prior methods들의 CHIAR/BLEU,POPE,MME 평가를 위한 출력 캡션을 생성하게 함.

### CHAIR/BLEU EVALUATION
생성한 CHIAR/BLEU 캡션 jsonl파일을 [eval/test_folder의 절대경로] 아래에 넣고,
python eval/caption_to_chair2.py --gt-caption-path [coco2014/annotations/captions_val2014.json의 절대경로] -c [eval/test_folder의 절대경로]
를 통해서 _chair.json 파일을 생성. 그 파일의 경로를 [chair_path]라고 하자.
python eval/eval_hallucination.py -v --metric chair --chair_input_path [chair_path]
를 통해 평가 

### POPE EVALUATION
pope캡션이 생성된 경로에 평가 결과가 함께 저장되어 있습니다. 

### MME EVALUATION
생성된 mme캡션들을 저장한 폴더 위치를 [mme_path]라고 하자.
python eval/mme_tool/calculation.py --results_dir [MAIN_CODES/eval/mme_tool/my_final_results의 절대경로] --captions_dir [mme_path]
를 통해 평가

### License
This repository is under BSD 3-Clause License. Many codes are based on Lavis with BSD 3-Clause License here, and 
https://github.com/BillChan226/HALC.