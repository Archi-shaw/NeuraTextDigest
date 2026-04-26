from typing import List, Dict, Any
from transformers import AutoTokenizer, pipeline
from src.textSummarizer.config.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path
        )

        self.pipe = pipeline(
            task="summarization",
            model=str(self.config.model_path),
            tokenizer=self.tokenizer
        )

    def predict(self, text: str) -> str:
        gen_kwargs = {
            "length_penalty": 0.8,
            "num_beams": 8,
            "max_length": 128
        }

        print("Dialogue:")
        print(text)

        result = self.pipe(text, **gen_kwargs)

        if not isinstance(result, list) or len(result) == 0:
            raise ValueError("Unexpected pipeline output")

        output = result[0].get("summary_text", "")
        print("\nModel Summary:")
        print(output)

        return output