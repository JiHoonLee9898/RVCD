import json

# 파일 경로 설정
input_text_file = "/home/onomaai/deeptext_multicaption/jihoon/eval_tool/Your_Results/existence.txt"
input_json_file = "/home/onomaai/deeptext_multicaption/jihoon/testing/MAIN_CODES/generated_captions/mme/llava-1.5/a1.0_b0.1_202501081637_llava-1.5__rvcd_seed_98_max_tokens_128_generated_captions.json"
output_file = "my_final_results/existence.txt"

# JSON 파일 로드
def load_json(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]

# 텍스트 파일 로드
def load_text_file(filename):
    with open(filename, "r") as f:
        return f.readlines()

# 응답 데이터를 딕셔너리로 정리
def process_json_data(json_data):
    response_dict = {}
    for entry in json_data:
        image_id = entry["image_id"]
        caption = entry["caption"].strip()
        if image_id not in response_dict:
            response_dict[image_id] = []
        response_dict[image_id].append(caption)
    return response_dict

# 텍스트 파일 확장하기
def expand_text_file(input_lines, response_dict):
    expanded_lines = []
    for line in input_lines:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        image_name, question, ground_truth = parts
        image_id = int(image_name.split(".")[0])
        model_response = response_dict.get(image_id, ["Unknown"]).pop(0)
        expanded_lines.append(f"{image_name}\t{question}\t{ground_truth}\t{model_response}\n")
    return expanded_lines

# 메인 실행부
if __name__ == "__main__":
    # 파일 로드
    json_data = load_json(input_json_file)
    text_data = load_text_file(input_text_file)

    # JSON 데이터 정리
    response_dict = process_json_data(json_data)

    # 텍스트 파일 확장
    expanded_lines = expand_text_file(text_data, response_dict)

    # 결과 저장
    with open(output_file, "w") as f:
        f.writelines(expanded_lines)

    print(f"Expanded results saved to {output_file}")
