#CUDA_VISIBLE_DEVICES=0 python main.py --model Qwen2-72B-Instruct \
#--base_url http://127.0.0.1:7777/v1/ \
#--dataset hitab \
#--qa_path dataset/hitab/test_samples.jsonl \
#--table_folder dataset/hitab/raw/ \
#--embedder_path gte-base \
#--embed_cache_dir dataset/hitab/ \
#--temperature 0.0 \
#--max_iteration_depth 4 \
#--seed 42
#
#
#CUDA_VISIBLE_DEVICES=2 python main.py \
#--model /home/amax/public/lyh/Qwen1.5-7B-Chat \
#--base_url http://218.192.110.170:8000/v1 \
#--dataset hitab \
#--qa_path dataset/hitab/test_samples.jsonl \
#--table_folder dataset/hitab/raw/ \
#--embedder_path /home/amax/public/lyh/GraphOTTER/bge-m3-sbert \
#--embed_cache_dir dataset/hitab/cache \
#--temperature 0.0 \
#--max_iteration_depth 4 \
#--seed 42 \
#--start 0 \
#--end 2

#CUDA_VISIBLE_DEVICES=0,1 python main.py \
#--model qwen-max \
#--base_url https://api.vveai.com/v1 \
#--dataset hitab \
#--qa_path dataset/hitab/test_samples.jsonl \
#--table_folder dataset/hitab/raw/ \
#--embedder_path /home/amax/public/lyh/GraphOTTER/gte-base \
#--embed_cache_dir dataset/hitab/cache \
#--temperature 0.0 \
#--max_iteration_depth 4 \
#--seed 42 \
#--start 1200 \
#--end 1500

#CUDA_VISIBLE_DEVICES=0,1 python main.py \
#--model qwen-max \
#--base_url https://api.vveai.com/v1 \
#--dataset hitab \
#--qa_path dataset/hitab/test_samples.jsonl \
#--table_folder dataset/hitab/raw/ \
#--embedder_path /home/amax/public/lyh/GraphOTTER/gte-base \
#--embed_cache_dir dataset/hitab/cache \
#--temperature 0.0 \
#--max_iteration_depth 4 \
#--seed 42 \
#--start 0 \
#--end 10 \
#--output_file dataset/hitab/output/-10.log

CUDA_VISIBLE_DEVICES=0,1 python main.py \
--model qwen-max-2025-01-25 \
--base_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
--dataset hitab \
--qa_path dataset/hitab/test_samples.jsonl \
--table_folder dataset/hitab/raw/ \
--embedder_path /home/amax/public/lyh/GraphOTTER/bge-m3-sbert \
--embed_cache_dir dataset/hitab/cache \
--temperature 0.0 \
--max_iteration_depth 4 \
--seed 42 \
--start 1200 \
--end 1500 \
--output_file dataset/hitab/outputQwen-max/202512016_BGE+reranker_output1200-1500.log \
--reranker_path /home/amax/public/lyh/GraphOTTER/bge-reranker-base