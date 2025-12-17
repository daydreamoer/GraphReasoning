#
# from configs import GEMINI_KEYS,hyperparameter
# from GraphRetriever.graph_retriver import GraphRetriever
#
# from Generator.openai_api import ChatGPTTool
# from GraphRetriever.dense_retriever import load_dense_retriever
# from Generator.Gemini_model import GeminiTool
# # from vertexai.preview import tokenization
# import argparse
#
# import tiktoken
# from dashscope import get_tokenizer
# from iterative_reasoning import GraphReasoner
# from compute_score import eval_ex_match,LLM_eval
# import json
# import jsonlines
#
# global tokenizer
# logger = None
#
# def augments():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument("--model", type=str, default='')
#     parser.add_argument("--base_url", type=str, default='')
#     parser.add_argument("--key", type=str,default='key')
#     parser.add_argument("--dataset", type=str, default='')
#     parser.add_argument("--qa_path", type=str, required=True)
#     parser.add_argument("--table_folder", type=str, )
#
#     parser.add_argument("--max_iteration_depth", type=int, required=True)
#     parser.add_argument("--start", required=True, type=int,default=0)
#     parser.add_argument("--end", required=True, type=int,default=20)
#
#     parser.add_argument("--temperature", type=float, default=0.0)
#     parser.add_argument("--seed", type=int, default=42)
#
#     parser.add_argument("--embed_cache_dir", type=str)
#     parser.add_argument("--embedder_path", type=str,default='')
#
#     parser.add_argument("--debug", type=bool, default=False)
#     args = parser.parse_args()
#
#     return args
#
# def load_model(args):
#     if 'gemini' in args.model:
#         gemini_key_index = 0
#         gemini_key = GEMINI_KEYS[gemini_key_index]
#         model = GeminiTool(gemini_key, args)
#     else:
#         model = ChatGPTTool(args)
#
#     dense_retriever = load_dense_retriever(args)
#
#     return  model,dense_retriever
#
# def load_data(args):
#
#     querys,answers,table_captions,tables,table_paths = [],[],[],[],[]
#
#     if args.dataset in ('hitab','Hitab'):
#         qas = []
#         with open(args.qa_path, "r+", encoding='utf-8') as f:
#             for item in jsonlines.Reader(f):
#                 qas.append(item)
#         qas = qas[args.start:args.end]
#
#         for qa in qas:
#             table_path = args.table_folder + qa['table_id'] + '.json'
#             with open(table_path, "r+", encoding='utf-8') as f:
#                 table = json.load(f)
#             table_captions.append(table['title'])
#             answers.append('|'.join([str(i) for i in qa['answer']]))
#             querys.append( qa['question'])
#             table_paths.append(table_path)
#             tables.append(table['texts'])
#     elif args.dataset in ('AIT-QA','ait-qa'):
#         with open(args.qa_path, 'r', encoding='utf-8') as f:
#             qas = json.load(f)
#         qas = qas[args.start:args.end]
#
#         for qa in qas:
#             tables.append(qa['table'])
#             answers.append('|'.join([str(i) for i in qa['answers']]))
#             querys.append( qa['question'])
#             table_captions.append('')
#             table_paths.append(qa)
#     return querys,answers,table_captions,tables,table_paths
#
#
# def main():
#     args = augments()
#
#     model, dense_retriever = load_model(args)
#     querys, answers, table_captions, tables,table_paths = load_data(args)
#
#     total_num,EM,LLM_EVAL = 0,0,0
#     for query,answer,caption,table,table_path in zip(querys, answers, table_captions, tables,table_paths):
#         unsafe = False
#         error = 3
#         graph_retriever = GraphRetriever(table_path, model, dense_retriever, args.dataset, table_cation=caption)
#         graph_reasoner = GraphReasoner(args, model, query, table, caption, graph_retriever, args.dataset)
#
#         output = graph_reasoner.iterative_reasoning()
#         # while error >0:
#         #     try:
#         #         output = graph_reasoner.iterative_reasoning()
#         #         break
#         #     except UserWarning as v:
#         #         print(query+ '\t' + '\t'+args.dataset+ '\t' +'不满足安全协议' + v.__str__()) # 没有通过gemini 的安全协议
#         #         unsafe = True
#         #         break
#         #     except Exception as e:
#         #         print(query+ '\t' +'报错了'+ '\t' +e.__str__())
#         #         error -= 1
#         #         continue
#         # if unsafe or error <= 0:
#         #     continue
#
#         print('模型回答为',output,'答案为',answer)
#         total_num += 1
#         EM += eval_ex_match(output,answer)
#         # LLM_EVAL += LLM_eval(model,query,output,answer)
#     print('EM:',EM/total_num)
#     # print('LLM EVAL:',LLM_EVAL/total_num)
#
#
#
#
# if __name__ == '__main__':
#     main()

import os
import sys
import argparse
import json
import jsonlines
import contextlib
import traceback

from configs import GEMINI_KEYS, hyperparameter
from GraphRetriever.graph_retriever import GraphRetriever
from Generator.openai_api import ChatGPTTool
from GraphRetriever.dense_retriever import load_dense_retriever
from Generator.Gemini_model import GeminiTool
from iterative_reasoning import GraphReasoner
from compute_score import eval_ex_match, LLM_eval
from sentence_transformers import CrossEncoder

global tokenizer
logger = None


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--base_url", type=str, default='')
    parser.add_argument("--key", type=str, default='key')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--qa_path", type=str, required=True)
    parser.add_argument("--table_folder", type=str)

    parser.add_argument("--max_iteration_depth", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=20)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embed_cache_dir", type=str)
    parser.add_argument("--embedder_path", type=str, default='')

    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--output_file", type=str, default='')

    # 2025.11添加reranker
    parser.add_argument("--reranker_path", type=str, default=None, help="Path to the bge-reranker-base model directory")

    args = parser.parse_args()

    return args


def load_model(args):
    if 'gemini' in args.model:
        gemini_key_index = 0
        gemini_key = GEMINI_KEYS[gemini_key_index]
        model = GeminiTool(gemini_key, args)
    else:
        model = ChatGPTTool(args)

    dense_retriever = load_dense_retriever(args)
    # --- 新增：加载 Reranker ---
    reranker_model = None
    if args.reranker_path:
        print(f"--- [Main] Loading Reranker from: {args.reranker_path} ---")
        # 假设 reranker_path 指向 bge-reranker-base 目录
        reranker_model = CrossEncoder(args.reranker_path)

    return model, dense_retriever, reranker_model


def load_data(args):
    querys, answers, table_captions, tables, table_paths = [], [], [], [], []

    if args.dataset.lower() == 'hitab':
        qas = []
        with open(args.qa_path, "r+", encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                qas.append(item)
        qas = qas[args.start:args.end]

        for qa in qas:
            table_path = os.path.join(args.table_folder, qa['table_id'] + '.json')
            with open(table_path, "r+", encoding='utf-8') as f:
                table = json.load(f)
            table_captions.append(table['title'])
            answers.append('|'.join([str(i) for i in qa['answer']]))
            querys.append(qa['question'])
            table_paths.append(table_path)
            tables.append(table['texts'])

    elif args.dataset.lower() == 'ait-qa':
        with open(args.qa_path, 'r', encoding='utf-8') as f:
            qas = json.load(f)
        qas = qas[args.start:args.end]

        for qa in qas:
            tables.append(qa['table'])
            answers.append('|'.join([str(i) for i in qa['answers']]))
            querys.append(qa['question'])
            table_captions.append('')
            table_paths.append(qa)

    return querys, answers, table_captions, tables, table_paths


def main():
    args = arguments()
    output_dir = os.path.join("dataset", args.dataset.lower(), "output")
    os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, "BGEoutput1200-1500.log")
    # 修改为在start.sh中指定output_file
    output_file = args.output_file if args.output_file else os.path.join(output_dir, "outputExcp.log")

    # with open(output_file, "w", encoding='utf-8') as f:
    #     tee = Tee(sys.stdout, f)
    #     with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
    #
    #         model, dense_retriever = load_model(args)
    #         querys, answers, table_captions, tables, table_paths = load_data(args)
    #
    #         total_num, EM, LLM_EVAL = 0, 0, 0
    #         for query, answer, caption, table, table_path in zip(querys, answers, table_captions, tables, table_paths):
    #             unsafe = False
    #             error = 3
    #             graph_retriever = GraphRetriever(table_path, model, dense_retriever, args.dataset, table_cation=caption)
    #             graph_reasoner = GraphReasoner(args, model, query, table, caption, graph_retriever, args.dataset)
    #
    #             output = graph_reasoner.iterative_reasoning()
    #             print('模型回答为:', output, '\t答案为:', answer)
    #
    #             total_num += 1
    #             EM += eval_ex_match(output, answer)
    #             LLM_EVAL += LLM_eval(model, query, output, answer)
    #
    #         print('EM:', EM / total_num if total_num > 0 else 0)
    #         print('LLM EVAL:', LLM_EVAL / total_num if total_num > 0 else 0)
    with open(output_file, "w", encoding='utf-8') as f:
        tee = Tee(sys.stdout, f)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):

            model, dense_retriever, reranker_model = load_model(args)
            querys, answers, table_captions, tables, table_paths = load_data(args)

            total_num, success_num, fail_num, EM, LLM_EVAL = 0, 0, 0, 0, 0

            for i, (query, answer, caption, table, table_path) in enumerate(
                    zip(querys, answers, table_captions, tables, table_paths)):
                print(f"\n=== 处理第 {i + 1} 条数据 ===")

                try:
                    graph_retriever = GraphRetriever(table_path, model, dense_retriever, args.dataset,
                                                     table_cation=caption, reranker_model=reranker_model)
                    graph_reasoner = GraphReasoner(args, model, query, table, caption, graph_retriever, args.dataset)

                    output = graph_reasoner.iterative_reasoning()
                    print('模型回答为:', output, '\t答案为:', answer)

                    total_num += 1
                    success_num += 1
                    EM += eval_ex_match(output, answer)
                    LLM_EVAL += LLM_eval(model, query, output, answer)

                except Exception as e:
                    print(f"处理第 {i + 1} 条数据时发生错误:")
                    print(traceback.format_exc())

                    output = "ERROR"
                    print('模型回答为:', output, '\t答案为:', answer)

                    total_num += 1
                    fail_num += 1
                    # 错误情况下，EM和LLM_EVAL计为0
                    EM += 0
                    LLM_EVAL += 0

            # 输出统计结果（多行显示）
            print("\n" + "=" * 50)
            print("处理结果统计:")
            print(f"总处理数: {total_num}")
            print(f"成功数: {success_num}")
            print(f"失败数: {fail_num}")
            if total_num > 0:
                print(f"成功率: {success_num / total_num * 100:.2f}%")
                print(f"EM: {EM / total_num:.4f}")
                print(f"LLM EVAL: {LLM_EVAL / total_num:.4f}")
            else:
                print("成功率: 0%")
                print("EM: 0")
                print("LLM EVAL: 0")
            print("=" * 50)


if __name__ == '__main__':
    main()
