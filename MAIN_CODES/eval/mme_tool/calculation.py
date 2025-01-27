import os, json
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='./my_final_results', type=str)
parser.add_argument('--captions_dir', default='/home/donut2024/JIHOON/MAIN_CODES/generated_captions/mme/llava-1.5', type=str)



eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}
eval_type_dict = {
    "Perception": ["existence", "count", "position", "color"],
    # "Cognition": []
}


class calculate_metrics:
    def divide_chunks(self, l, n=2):
        # looping till length l
        for i in range(0, len(l), n): 
            yield l[i:i + n]
        
        return 

    def parse_pred_ans(self, pred_ans):
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]

            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"

        return pred_label


    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "other": -1,
        }
        
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds) 

        clean_gts = []
        clean_preds = []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        return metric_dict


    def process_result(self, results_dir):

        model_score_dict = dict()
        for eval_type, task_name_list in eval_type_dict.items():
            print("===========", eval_type, "===========")
           
            scores = 0
            task_score_dict = dict()

            for task_name in task_name_list:

                task_txt = os.path.join(results_dir, task_name + ".txt")
                lines = open(task_txt, 'r').readlines()
                chunk_lines = list(self.divide_chunks(lines)) # one image corresponds to two questions
                
                img_num = len(chunk_lines)
                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []

                for img_items in chunk_lines:
                    assert len(img_items) == 2
                    img_correct_num = 0

                    for img_item in img_items:
                        img_name, question, gt_ans, pred_ans = img_item.split("\t")

                        gt_ans = gt_ans.lower()
                        pred_ans = pred_ans.lower()

                        assert gt_ans in ["yes", "no"] # gt can only be yes or no.

                        pred_ans = self.parse_pred_ans(pred_ans)
                        assert pred_ans in ["yes", "no", "other"]

                        gts.append(gt_ans)
                        preds.append(pred_ans)
                        
                        if gt_ans == pred_ans:
                            img_correct_num += 1
                        
                        if pred_ans not in ["yes", "no"]:
                            task_other_ans_num += 1

                    if img_correct_num == 2:
                        acc_plus_correct_num += 1

                # cal TP precision acc, etc.
                metric_dict = self.compute_metric(gts, preds)
                acc_plus = acc_plus_correct_num / img_num
                metric_dict["acc_plus"] = acc_plus
                
                
                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v*100
                
                task_score_dict[task_name] = task_score
                
                scores += task_score

            print("total score:", scores, "\n")
            for task_name, score in task_score_dict.items():
                print("\t", task_name, " score:", score)
            print("\n")
        
        return 



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


if __name__ == "__main__":
    # 폴더 경로
    args = parser.parse_args()
    input_json_files_folder = args.captions_dir  # 폴더 경로

    # 폴더 안의 모든 파일을 리스트에 담기
    all_files = [os.path.join(input_json_files_folder, file) for file in os.listdir(input_json_files_folder) if os.path.isfile(os.path.join(input_json_files_folder, file))]

    for input_json_file in all_files:
        now_mme_type = None
        for mme_type in ["existence", "count", "position", "color"]:
            if mme_type in input_json_file:
                now_mme_type = mme_type
                break
        input_text_file = f'./eval/mme_tool/Your_Results/{now_mme_type}.txt'
        output_file = f"./eval/mme_tool/my_final_results/{now_mme_type}.txt"

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

    cal = calculate_metrics()
    
    results_dir = args.results_dir
    cal.process_result(results_dir)


