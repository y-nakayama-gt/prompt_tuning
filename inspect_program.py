import dspy
from dotenv import load_dotenv
import os

load_dotenv()

llm = dspy.LM(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

dspy.configure(lm=llm)


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
    def __init__(self) -> None:
        super().__init__()

        self.zunda_module = dspy.ChainOfThought(ZundaSinature)
        self.mochi_module = dspy.Predict(MochiSignature)

    def forward(self, question: str, lm: dspy.LM | None = None) -> dspy.Prediction:
        zunda_answer = self.zunda_module(question=question, lm=lm).zunda_answer
        return self.mochi_module(text=zunda_answer, lm=lm)


program = dspy.load("./results/", allow_pickle=True)

separator = "=" * 60

for name, predictor in program.named_predictors():
    print(f"\n{separator}")
    print(f"Predictor: {name}")
    print(separator)

    print("\n--- Instructions (最適化されたシステムプロンプト) ---")
    print(predictor.signature.instructions)

    print("\n--- Demos (few-shotデモ) ---")
    if predictor.demos:
        for i, demo in enumerate(predictor.demos):
            print(f"\n[Demo {i + 1}]")
            for key, value in demo.items():
                print(f"  {key}: {value}")
    else:
        print("  (デモなし)")
