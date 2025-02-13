
To download the datasets used to pre-train relik on djlama: 
```
klim@djlama:~/storage/relik/data/GENRE$ 
bash scripts_genre/download_all_datasets.sh
```

To download relik index to djlama: 
```
data/klim/repositories/wikidata-temp/benchmarks/models/relik/index$ 
wget https://huggingface.co/sapienzanlp/relik-retriever-e5-base-v2-aida-blink-wikipedia-index/resolve/main/documents.jsonl
```

To preprocess nyt on djlama: 
```
klim@djlama:/data/klim/repositories/relik$ 
python scripts/data/nyt/preprocess_nyt.py \
    /home/klim/storage/relik/data/raw_nyt data/nyt/processed/
```

Executing relik on djlama from the following directory /data/klim/repositories/relik : 
```
python scripts/data/blink/preprocess_genre_blink.py \
/home/klim/storage/relik/data/GENRE/datasets/blink-dev-kilt.jsonl \
/home/klim/storage/relik/data/blink/processed/blink-dev-kilt-relik.jsonl
```
and now executing for train: 
```
python scripts/data/blink/preprocess_genre_blink.py \
/home/klim/storage/relik/data/GENRE/datasets/blink-train-kilt.jsonl \
/home/klim/storage/relik/data/blink/processed/blink-train-kilt-relik.jsonl
```

13.02.2025 - Now executing the creation of the windows for train set: 
```
PYTHONPATH=. python scripts/data/create_windows.py /home/klim/storage/relik/data/blink/processed/blink-train-kilt-relik.jsonl /home/klim/storage/relik/data/blink/processed/blink-train-kilt-relik-windowed.jsonl
```

13.02.2025 - Now executing the creation of the windows for dev set: 
```
PYTHONPATH=. python scripts/data/create_windows.py /home/klim/storage/relik/data/blink/processed/blink-dev-kilt-relik.jsonl /home/klim/storage/relik/data/blink/processed/blink-dev-kilt-relik-windowed.jsonl
```

13.02.2025 - Now executing the conversion to dpr script on djlama: 
```
PYTHONPATH=. python scripts/data/retriever/convert_to_dpr.py \
  /home/klim/storage/relik/data/blink/processed/blink-train-kilt-relik-windowed.jsonl \
  /home/klim/storage/relik/data/blink/processed/blink-train-kilt-relik-windowed-dpr.jsonl \
  index/documents.jsonl
```

Working on ```s01_test_relik.py```
