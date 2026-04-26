from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import evaluate
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from src.textSummarizer.entity import ModelEvaluationConfig




class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    
    def generate_batch_sized_chunks(self,list_of_elements, batch_size):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    
    def calculate_metric_on_test_ds(self,dataset, metric, model, tokenizer, 
                               batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu", 
                               column_text="article", 
                               column_summary="highlights"):
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):
            
            inputs = tokenizer(article_batch, max_length=1024,  truncation=True, 
                            padding="max_length", return_tensors="pt")
            
            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device), 
                            length_penalty=0.8, num_beams=8, max_length=128)
            ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''
            
            # Finally, we decode the generated texts, 
            # replace the  token, and add the decoded texts with the references to the metric.
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=True) 
                for s in summaries]      
            
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
            
            
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
            
        #  Finally compute and return the ROUGE scores.
        score = metric.compute()
        return score


    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _load_pretrained(loader, path, **kwargs):
            p = Path(path)
            if p.exists():
                try:
                    return loader.from_pretrained(str(p), local_files_only=True, **kwargs)
                except Exception:
                    return loader.from_pretrained(str(p), **kwargs)
            # If local path doesn't exist, try loading from hub (repo id)
            try:
                return loader.from_pretrained(path, **kwargs)
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not load from local path '{path}' and remote load failed: {e}. "
                    "Make sure the path is correct or provide a valid Hugging Face repo id like 'username/repo'."
                )

        tokenizer = _load_pretrained(AutoTokenizer, self.config.tokenizer_path)
        model_pegasus = _load_pretrained(AutoModelForSeq2SeqLM, self.config.model_path).to(device)
       
        #loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)


        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        rouge_metric = evaluate.load("rouge")

        score = self.calculate_metric_on_test_ds(
            dataset_samsum_pt['test'][0:10], rouge_metric, model_pegasus, tokenizer,
            batch_size=2, column_text='dialogue', column_summary='summary'
        )

        def _extract_fmeasure(v):
            try:
                return float(v.mid.fmeasure)
            except Exception:
                try:
                    return float(v['mid']['fmeasure'])
                except Exception:
                    try:
                        return float(v['fmeasure'])
                    except Exception:
                        return float(v)

        rouge_dict = {rn: _extract_fmeasure(score[rn]) for rn in rouge_names}

        df = pd.DataFrame(rouge_dict, index=['pegasus'])
        df.to_csv(self.config.metric_file_name, index=False)

        