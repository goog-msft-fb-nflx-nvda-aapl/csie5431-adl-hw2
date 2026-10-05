# ADL HW2: RAG

Use Python 3.12.9.

1. Download the dataset and place it under `data/` so that each split
   contains `dialogues.jsonl` and `products.jsonl`.

2. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Implement retrieval and reranking only inside
   `retrieval_utils.py`, then run `retrieval.py` to generate the ranked results:

   ```bash
   python retrieval.py --dialogues "dialogues_path" --products "products_path" --output "output_path"
   ```

4. Evaluate the results:

   ```bash
   python evaluate.py --retrieval-results "path_to_retrieval_result" --dialogues "dialogues_path" --products "products_path" --output "output_path"
   ```
