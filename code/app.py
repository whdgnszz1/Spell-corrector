from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
from utils import post_process, find_closest_candidate

app = FastAPI()

# 모델과 토크나이저 로드
model_path = "./models/checkpoint-8500_latest"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 후보군 데이터 로드
candidate_file = "../datasets/dataset_candidate_case3.json"
with open(candidate_file, 'r') as f:
    json_dataset = json.load(f)
candidates = [data['annotation']['cor_sentence'] for data in json_dataset['data']]

# 입력 데이터 모델 정의
class TextInput(BaseModel):
    text: str

# 텍스트 교정 엔드포인트
@app.post("/correct")
async def correct_text(input: TextInput):
    try:
        # 입력 문장 토큰화
        tokenized = tokenizer(input.text, return_tensors="pt", max_length=128, truncation=True)
        input_ids = tokenized["input_ids"].to(device)

        # 모델로 문장 생성
        with torch.no_grad():
            res = model.generate(
                inputs=input_ids,
                num_beams=10,
                num_return_sequences=1,
                temperature=0.7,
                repetition_penalty=2.0,
                length_penalty=1.0,
                no_repeat_ngram_size=2,
                max_length=input_ids.size()[1] + 5
            ).cpu().tolist()[0]

        # 생성된 문장 디코딩
        raw_prd_sentence = tokenizer.decode(res, skip_special_tokens=True).strip()

        # 후처리: 반복 단어 제거 및 단어 수 조정
        post_processed_prd_sentence = post_process(raw_prd_sentence, input.text)
        final_prd_sentence = find_closest_candidate(post_processed_prd_sentence, candidates)

        # 결과 반환
        return {
            "corrected_text": final_prd_sentence,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)