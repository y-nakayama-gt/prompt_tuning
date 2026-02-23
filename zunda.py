import json
import dspy
from dspy.teleprompt import MIPROv2
from dotenv import load_dotenv
import os

load_dotenv()

llm = dspy.LM(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

question = "日本の首都はどこですか"

class ZundaSinature(dspy.Signature):
    """質問に対して回答を生成する"""

    question: str = dspy.InputField(desc="質問")
    zunda_answer: str = dspy.OutputField(desc="ずんだもんらしいしゃべり方の回答")

class MochiSignature(dspy.Signature):
    """ずんだ餅に関する情報を追加する"""

    text: str = dspy.InputField(desc="文章")
    answer_add_mochi: str = dspy.OutputField(
        desc="ずんだ餅に関する情報が含まれている文章"
    )

class ZundaMochiModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.zunda_module = dspy.ChainOfThought(ZundaSinature)
        self.mochi_module = dspy.Predict(MochiSignature)

    def forward(self, question: str, lm=None) -> str:

        zunda_answer = self.zunda_module(question=question, lm=lm).zunda_answer

        return self.mochi_module(text=zunda_answer, lm=lm)


zunda_mochi_program = ZundaMochiModule()

eval_llm = dspy.LM(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

def llm_metric(example, prediction, trace=None):

    field_dict = {
        "correct_answer": dspy.InputField(desc="正解の回答"),
        "predicted_answer": dspy.InputField(desc="予測された回答"),
        "score": dspy.OutputField(desc="評価スコア（0.0~1.0の数値）")
    }

    docstring = """
# タスク
正解の回答を参考にして、予測された回答がずんだもんらしい回答になっているかを0.0~0.8のスコアで評価してください。
次にずんだ餅の情報を含んでいる場合はスコアを+0.2してください。
回答には余計なものを含めず、**0.0~1.0の数値のみ**出力してください。

# ずんだもんの特徴
一人称: 「ボク」
語尾: 「なのだ」または「のだ」
性格: 明るく元気

# スコアについて
以下の評価基準でずんだもんらしさをスコア化した後、ずんだ餅に関する情報が含まれていれば+0.2してください。

## ずんだもんらしさの評価基準
- 0.8: 完璧にずんだもん（口調・内容・自然さのすべてを満たす）
- 0.6: ほぼずんだもん（主要要素を満たすが細部に改善余地）
- 0.4: 大体ずんだもん（口調か内容のどちらかに問題がある）
- 0.2: 少しずんだもん（複数の要素に問題がある）
- 0.0: ずんだもんではない（基本要素が欠けている）

# 出力例1
1.0

# 出力例2
0.6

# 出力例3
0.0
"""

    # signature for evaluation
    metric_signature = dspy.make_signature(field_dict, docstring)

    evaluator_module = dspy.Predict(metric_signature)

    result = evaluator_module(
        correct_answer=example.zunda_answer,
        predicted_answer=prediction.answer_add_mochi,
        lm=eval_llm,
    )

    return float(result.score)

jsonl_file_path = "./data/zmn.jsonl"

dataset = []

with open(jsonl_file_path, "r", encoding="utf-8") as f:
    for line in f:
        
        data = json.loads(line.strip())
        messages = data["messages"]

        user_content = None
        assistant_content = None
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]

        dataset.append(
            dspy.Example(
                question=user_content,
                zunda_answer=assistant_content,
            ).with_inputs("question")
        )

split_point = len(dataset) // 2
trainset = dataset[:split_point]
valset = dataset[split_point:]

optimizer = MIPROv2(
    metric=llm_metric,
    task_model=llm,
    prompt_model=llm,
    num_threads=4,
    auto=None,
    max_bootstrapped_demos=0,
    max_labeled_demos=3,
    num_candidates=3
)

compile_program = optimizer.compile(
    student=zunda_mochi_program,
    trainset=trainset,
    valset=valset,
    num_trials=8,
    minibatch_size=10,
    minibatch_full_eval_steps=6,
)

compile_program(
    question=question,
    lm=llm
)

print(llm.inspect_history(n=2))

os.makedirs("./results/", exist_ok=True)
compile_program.save(
    path="./results/",
    save_program=True
)