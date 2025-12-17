from GraphRetriever.table_reader import hitab_table_converter, ait_qa_converter
from collections import OrderedDict

from GraphRetriever.BM25_Retriever import EM_retriever
from configs import PROMPT_TEMPLATE, GEMINI_KEYS, hyperparameter
import ast
import copy

import os
from Generator.parse_output import remove_quotes
import time
import re
import random
from tools import table2Tuple

table_loader_dict = {
    'hitab': hitab_table_converter,
    'ait-qa': ait_qa_converter,

}


class GraphRetriever:
    def __init__(self, table_path, model, dense_retriever, dataset, table_cation='', reranker_model=None):

        self.dealed_rows, self.dealed_cols, \
            self.rows, self.cols, self.merged_cells = table_loader_dict[dataset](table_path)
        self.table_cation = table_cation
        self.cells = [str(c) for r in self.rows for c in r]
        self.dealed_cells = [str(c) for r in self.dealed_rows for c in r]
        self.cell_ids = [i for i in range(len(self.rows) * len(self.rows[0]))]

        self.LLM_model = model
        self.reranker_model = reranker_model  # <-- 新增：保存 reranker 实例
        self.dataset = dataset
        self.dense_retriever = dense_retriever
        if dataset in ('ait-qa'):
            file_name = table_path['table_id']
        else:
            file_name = os.path.split(os.path.splitext(table_path)[0])[1]
        self.dense_retriever.load_graph(self.cells, dataset, file_name)

    def get_neighbors(self, add_id_list, get_same_row, get_all_nei=False):
        match_id_list = add_id_list
        cell_topk, _ = self.tableId2Content(add_id_list, self.dealed_rows)

        # 获取命中单元格同行/列单元格
        hit_cell_same_row_col = self.getSameRow_ColCells(match_id_list)
        same_col_cells_id = hit_cell_same_row_col[add_id_list[0]]['same_col_cells_id']
        same_row_cells_id = hit_cell_same_row_col[add_id_list[0]]['same_row_cells_id']
        hit_cell_neighbors = {
            '同行': [],
            '同列': []
        }
        hit_cell_neighbors_content_id = OrderedDict()

        if get_same_row or get_all_nei:
            for row in same_row_cells_id:
                same_row_cells, same_row_cell_ids = self.tableId2Content(row, self.dealed_rows)
                same_row_cells_tuple = self.cell2Tuple(same_row_cell_ids)
                hit_cell_neighbors['同行'] += same_row_cells_tuple
                for k in range(len(same_row_cells)):
                    hit_cell_neighbors_content_id[same_row_cell_ids[k]] = same_row_cells[k]
        if get_all_nei or not get_same_row:
            for col in same_col_cells_id:
                same_col_cells, same_col_cell_ids = self.tableId2Content(col, self.dealed_cols)
                same_col_cells_tuple = self.cell2Tuple(same_col_cell_ids)
                hit_cell_neighbors['同列'] += same_col_cells_tuple
                for k in range(len(same_col_cells)):
                    hit_cell_neighbors_content_id[same_col_cell_ids[k]] = same_col_cells[k]
        result = []
        cell_topk_tuple = self.cell2Tuple(match_id_list)
        if get_same_row or get_all_nei:
            result.append(
                'The row containing "{}" includes the following nodes: {}.'.format(
                    cell_topk_tuple[0] if len(cell_topk_tuple) == 1 else cell_topk_tuple,
                    str(hit_cell_neighbors['同行'])))
        if get_all_nei or not get_same_row:
            result.append(
                'The column containing "{}" includes the following nodes: {}.'.format(
                    cell_topk_tuple[0] if len(cell_topk_tuple) == 1 else cell_topk_tuple,
                    str(hit_cell_neighbors['同列'])))
        result = '\n'.join(result)
        return cell_topk, result, hit_cell_neighbors_content_id

    # def search_cell(self, query, topk=3, LLM_match_id_list=[], select_cell=False):
    #     LLM_match_topk = [self.dealed_cells[i] for i in LLM_match_id_list]
    #     if query in self.cells:
    #         cell_topk, match_id_list, sparse_scores = EM_retriever(self.dealed_cells, query, isEM=True)
    #         cell_topk, match_id_list, cell_topk_tuple = self.cell2Tuple(match_id_list, add_merged_cells=True)
    #         return cell_topk, match_id_list, cell_topk_tuple
    #     else:
    #         if not select_cell:
    #             sparse_cell_topk, sparse_match_id_list, sparse_scores = EM_retriever(self.dealed_cells, query,
    #                                                                                  isEM=True)
    #         else:
    #             # sparse_cell_topk, sparse_match_id_list, sparse_scores = EM_retriever(self.dealed_cells, query)
    #             sparse_cell_topk, sparse_match_id_list, sparse_scores = [], [], None
    #
    #         dense_cell_topk, dense_match_id_list, dense_scores = self.dense_retriever.search_single(query, topk=20)
    #
    #         if not select_cell:
    #             cell_topk = sparse_cell_topk
    #             match_id_list = sparse_match_id_list
    #             if len(cell_topk) < topk:
    #                 for j in range(len(dense_cell_topk[:(topk - len(cell_topk))])):
    #                     if dense_scores[:(topk - len(cell_topk))][j] > hyperparameter[self.dataset]['dense_score']:
    #                         dense_cell = dense_cell_topk[:(topk - len(cell_topk))][j]
    #                         temp_cell_topk, temp_match_id, _ = EM_retriever(self.dealed_cells, dense_cell, isEM=True)
    #                         cell_topk += temp_cell_topk
    #                         match_id_list += temp_match_id
    #             cell_topk, match_id_list, cell_topk_tuple = self.cell2Tuple(match_id_list, add_merged_cells=True)
    #         else:
    #             dense_topk = topk
    #             temp_dense_topk, temp_dense_id_list = [], []
    #             for i in range(len(dense_match_id_list)):
    #                 if dense_topk <= 0:
    #                     break
    #                 dense_id = dense_match_id_list[i]
    #                 if dense_id in sparse_match_id_list or dense_id in LLM_match_id_list or dense_cell_topk[
    #                     i] in LLM_match_topk:
    #                     dense_topk -= 1
    #                 else:
    #                     temp_dense_topk.append(dense_cell_topk[i])
    #                     temp_dense_id_list.append(dense_match_id_list[i])
    #                     dense_topk -= 1
    #
    #             temp, temp_ = [], []
    #             for i in range(len(sparse_match_id_list)):
    #                 if sparse_match_id_list[i] not in LLM_match_id_list:
    #                     temp.append(sparse_match_id_list[i])
    #                     temp_.append(sparse_cell_topk[i])
    #             sparse_match_id_list, sparse_cell_topk = temp, temp_
    #
    #             temp = sparse_match_id_list + temp_dense_id_list
    #             hybird_cell_id = OrderedDict()
    #             hybird_cell = sparse_cell_topk + temp_dense_topk
    #             for i in range(len(temp)):
    #                 # if hybird_cell[i]:
    #                 hybird_cell_id[temp[i]] = hybird_cell[i]
    #             # cell_topk,match_id_list = self.LLM_reranker(query,hybird_cell_id ,topk)
    #             cell_topk, match_id_list = list(hybird_cell_id.values()), list(hybird_cell_id.keys())
    #             cell_topk_tuple = self.cell2Tuple(match_id_list)
    #         return cell_topk, match_id_list, cell_topk_tuple
    def search_cell(self, query, topk=4, LLM_match_id_list=[], select_cell=False):
        """
        [改进版] 混合检索：两路召回 + Reranker 重排序
        :param query: 用户问题
        :param topk: 最终需要的单元格数量
        :param LLM_match_id_list: LLM 已经选出的单元格ID (用于去重)
        :param select_cell: 是否为图初始化阶段 (如果是，则需排除 LLM 已选的结果)
        """

        # --- 步骤 1: 精确匹配 (Exact Match) 优先 ---
        # 这是一个极低成本的高效检查，保留原逻辑
        if query in self.cells:
            cell_topk, match_id_list, _ = EM_retriever(self.dealed_cells, query, isEM=True)
            cell_topk, match_id_list, cell_topk_tuple = self.cell2Tuple(match_id_list, add_merged_cells=True)
            return cell_topk, match_id_list, cell_topk_tuple

        # --- 步骤 2: 两路宽召回 (Recall) ---
        # 我们扩大召回数量 (N)，目的是不错过任何潜在相关项，交给 Reranker 去筛选
        candidate_N = 20

        # 2.1 稀疏召回 (Sparse / BM25)
        # 注意：这里我们忽略 scores，因为 BM25 分数和 Dense 分数不可比
        sparse_cells, sparse_ids, _ = EM_retriever(self.dealed_cells, query, isEM=True)
        sparse_ids = sparse_ids[:candidate_N]

        # 2.2 密集召回 (Dense / Vector)
        # 同样只取 Top N 个 ID
        _, dense_ids, _ = self.dense_retriever.search_single(query, topk=candidate_N)

        # --- 步骤 3: 合并候选集 (Union & Deduplicate) ---
        # 使用 set 进行去重合并
        candidate_ids_set = set(sparse_ids + dense_ids)

        # 过滤掉 LLM 已经选过的结果 (如果 select_cell=True)
        if select_cell and LLM_match_id_list:
            for existing_id in LLM_match_id_list:
                if existing_id in candidate_ids_set:
                    candidate_ids_set.remove(existing_id)

        candidate_ids = list(candidate_ids_set)

        # 如果候选集为空 (极罕见情况)，直接返回空
        if not candidate_ids:
            return [], [], []

        # --- 步骤 4: 重排序 (Rerank) ---
        final_top_ids = []

        if self.reranker_model is not None:
            # 4.1 准备输入数据: [(query, cell_content), ...]
            # 我们需要先根据 ID 获取内容
            candidates_content, _ = self.tableId2Content(candidate_ids, self.dealed_rows)
            rerank_pairs = [[query, str(content)] for content in candidates_content]

            # 4.2 调用 Reranker 打分
            # predict 返回的是 logits 或 scores
            scores = self.reranker_model.predict(rerank_pairs)

            # 4.3 组合 (ID, Score) 并排序
            id_score_pairs = list(zip(candidate_ids, scores))
            # 按分数降序排列
            id_score_pairs.sort(key=lambda x: x[1], reverse=True)

            # 提取排序后的 ID
            final_top_ids = [pid for pid, score in id_score_pairs]

        else:
            # 4.1 (回退策略) 如果没有 Reranker，使用 RRF (倒数排名融合)
            # 这是一个非常鲁棒的无参数融合方法
            rrf_scores = {}
            k_const = 60

            for rank, pid in enumerate(sparse_ids):
                rrf_scores[pid] = rrf_scores.get(pid, 0) + 1.0 / (k_const + rank + 1)

            for rank, pid in enumerate(dense_ids):
                rrf_scores[pid] = rrf_scores.get(pid, 0) + 1.0 / (k_const + rank + 1)

            # 只对在候选集中的 ID 进行排序
            valid_rrf = [(pid, rrf_scores[pid]) for pid in candidate_ids if pid in rrf_scores]
            valid_rrf.sort(key=lambda x: x[1], reverse=True)
            final_top_ids = [pid for pid, score in valid_rrf]

        # --- 步骤 5: 截断并返回 ---
        # 取最终的 Top-K
        final_top_ids = final_top_ids[:topk]

        # 调用辅助函数生成返回格式 (内容列表, ID列表, 元组列表)
        # 注意：这里我们使用 add_merged_cells=True 保持与原逻辑一致
        cell_topk, match_id_list, cell_topk_tuple = self.cell2Tuple(final_top_ids, add_merged_cells=True)

        return cell_topk, match_id_list, cell_topk_tuple

    def LLM_generate(self, prompt, system_instruction, isrepeated=0.0, response_format=None):
        # 复制 isrepeated 以避免修改原始值
        isrepeated_, error = copy.deepcopy(isrepeated), 3

        # 如果模型名称中没有 gemini，则将 prompt 转换为列表
        if 'gemini' not in self.LLM_model.model_name:
            prompt = [prompt]

        # --- [新增/修改]：针对 Gemini 的参数适配逻辑 ---
        # --- [修复]：正确构建参数字典 ---
        generate_kwargs = {}
        if 'gemini' in self.LLM_model.model_name:
            # 如果是 Gemini，且请求的是 JSON 格式，则转换为 response_mime_type
            if response_format and (
                    (isinstance(response_format, dict) and response_format.get("type") == "json_object") or
                    response_format == "application/json"
            ):
                generate_kwargs['response_mime_type'] = "application/json"
        else:
            # 对于非 Gemini 模型（如 GPT/DeepSeek），如果存在 response_format 则加入字典
            #前提是底层模型支持该参数，不支持会被下面的 try-except 捕获
            if response_format:
                generate_kwargs['response_format'] = response_format

        # -------------------------------------------
        # 循环尝试3次
        while error > 0:
            try:
                # 调用 generate 方法并传递必要的参数
                # select_id_text = self.LLM_model.generate(
                #     prompt,
                #     system_instruction=system_instruction,
                #     isrepeated=isrepeated_,
                #     response_format=response_format  # <-- 新增response_format参数
                # )
                ##需要模型支持格式控制参数
                select_id_text = self.LLM_model.generate(
                    prompt,
                    system_instruction=system_instruction,
                    isrepeated=isrepeated_,
                    **generate_kwargs  # 使用解包的 kwargs，而不是显式的 response_format
                )


                # 从返回文本中提取 JSON 格式的部分
                json_match = re.search(r'```json(.*?)```', select_id_text, re.DOTALL)
                json_match3 = re.search(r'\[.*\]', select_id_text)

                # 解析匹配到的 JSON 部分
                if json_match:
                    json_string = json_match.group(1)
                    try:
                        select_id = ast.literal_eval(json_string.strip().strip('\n'))
                    except Exception as e:
                        print(f"LLM输出解析报错: {select_id}, {e}")
                        raise Exception(f"LLM输出解析报错: {select_id}, {e}")

                # 如果输出是符合 JSON 数组格式的字符串
                elif select_id_text.strip('```').strip('"').strip("'").strip().startswith('[') and \
                        select_id_text.strip('```').strip('"').strip("'").strip().endswith(']'):
                    try:
                        select_id = ast.literal_eval(select_id_text.strip('```').strip('"').strip("'").strip())
                    except Exception as e:
                        print(f"LLM输出解析报错: {select_id}, {e}")
                        raise Exception(f"LLM输出解析报错: {select_id}, {e}")

                # 如果输出是符合 JSON 数组的字符串格式
                elif json_match3:
                    try:
                        json_str = json_match3.group()
                        select_id = ast.literal_eval(json_str)
                    except Exception as e:
                        print(f"LLM输出解析报错: {select_id}, {e}")
                        raise Exception(f"LLM输出解析报错: {select_id}, {e}")

                # 如果以上方法都无法正确解析，重新尝试
                else:
                    isrepeated_ = 0.7
                    error -= 1
                    continue

                # 如果 select_id 是有效的列表并且长度大于等于0，则返回
                if isinstance(select_id, list) and len(select_id) >= 0:
                    break
                else:
                    isrepeated_ = 0.7
                    error -= 1
                    continue

            # 捕获可能的异常
            except Exception as e:
                print(f"生成时发生错误: {e}")
                error -= 1
                time.sleep(2.0)

        # 如果超过最大重试次数仍然失败，则抛出警告
        if error <= 0:
            raise UserWarning("LLM无法提取cell")

        return select_id

    # def initialize_subgraph(self, query):
    #
    #     # --- 1. 原始选择 (保持不变) ---
    #     # LLM select first
    #     LLM_cell_topk, LLM_match_id_list = self.LLM_select_cells_from_table(query, hyperparameter[self.dataset][
    #         'LLM_select_cells'])
    #     # then retriever select
    #     retriever_cell_topk, retriever_match_id_list, _ = self.search_cell(query, topk=hyperparameter[self.dataset][
    #         'dense_topk'],
    #                                                                       LLM_match_id_list=LLM_match_id_list,
    #                                                                       select_cell=True)
    #
    #     # --- 2. 新增：重排序逻辑 (Reranking Logic) ---
    #
    #     # 2a. 合并 LLM 和 Retriever 的结果，并去重
    #     # (使用 OrderedDict 自动按 ID 去重，同时保留添加顺序)
    #     candidate_cells = OrderedDict()
    #     for i in range(len(LLM_match_id_list)):
    #         candidate_cells[LLM_match_id_list[i]] = LLM_cell_topk[i]
    #     for i in range(len(retriever_match_id_list)):
    #         # 如果 ID 已存在 (LLM选过)，则不会覆盖
    #         candidate_cells.setdefault(retriever_match_id_list[i], retriever_cell_topk[i])
    #
    #     # 2b. 准备用于图构建的 ID 列表。默认使用合并后的顺序。
    #     graph_build_id_list = list(candidate_cells.keys())
    #
    #     # 2c. 如果 Reranker 存在且有候选单元格，则执行重排序
    #     if self.reranker_model is not None and candidate_cells:
    #
    #         # 提取内容用于预测
    #         candidate_contents = list(candidate_cells.values())
    #
    #         # 创建 (query, content) 对
    #         rerank_input_pairs = [(query, content) for content in candidate_contents]
    #
    #         print(f"--- [GraphRetriever] Reranking {len(rerank_input_pairs)} candidates... ---")
    #
    #         # 2d. [核心] 调用 Reranker (CrossEncoder) 的 predict 方法获取分数
    #         scores = self.reranker_model.predict(rerank_input_pairs)
    #
    #         # 2e. 将 ID、内容和分数打包
    #         combined_data = list(zip(graph_build_id_list, candidate_contents, scores))
    #
    #         # 2f. 按分数(item[2])降序排序
    #         sorted_data = sorted(combined_data, key=lambda item: item[2], reverse=True)
    #
    #         # 2g. 解包，获取重排后的 ID 列表
    #         graph_build_id_list, _, sorted_scores = zip(*sorted_data)
    #         graph_build_id_list = list(graph_build_id_list) # 转换回 list
    #
    #         print(f"--- [GraphRetriever] Reranking complete. Top 3 IDs: {graph_build_id_list[:3]} (Scores: {list(sorted_scores)[:3]}) ---")
    #
    #
    #     # --- 3. 图构建 (使用重排后的列表) ---
    #
    #     # [修改]：使用重排后的 `graph_build_id_list`
    #     hit_cell_same_row_col = self.getSameRow_ColCells(graph_build_id_list)
    #
    #     # [修改]：此函数现在基于重排后的列表（通过 hit_cell_same_row_col）来构建图
    #     connect_graph, connect_id_cell_dict, connect_graph_result = self.hit_cell_connect_graph(hit_cell_same_row_col)
    #
    #
    #     # --- 4. 准备返回值 (保持原始结构不变) ---
    #     # (这部分代码保持不变，以确保下游 iterative_reasoning.py 的兼容性)
    #     # (它返回的是未经重排的、分离的 LLM/retriever 列表)
    #     cell_id_content_topk = OrderedDict(
    #         {
    #             'LLM_select': {},
    #             'retriever_select': {}
    #         }
    #     )
    #     cell_tuple_topk = copy.deepcopy(cell_id_content_topk)
    #
    #     for i in range(len(LLM_cell_topk)):
    #         cell_id_content_topk['LLM_select'][LLM_match_id_list[i]] = LLM_cell_topk[i]
    #     cell_tuple_topk['LLM_select'] = self.cell2Tuple(LLM_match_id_list)
    #     for i in range(len(retriever_cell_topk)):
    #         cell_id_content_topk['retriever_select'][retriever_match_id_list[i]] = retriever_cell_topk[i]
    #     cell_tuple_topk['retriever_select'] = self.cell2Tuple(retriever_match_id_list)
    #
    #     # --- 5. 返回 ---
    #     # [结果]：返回的 connect_graph_result 现在是基于重排后的 ID 列表生成的
    #     return cell_id_content_topk, cell_tuple_topk, connect_id_cell_dict, connect_graph_result

    def initialize_subgraph(self, query):
        # 1. LLM 初选 (语义直觉)
        LLM_cell_topk, LLM_match_id_list = self.LLM_select_cells_from_table(
            query, hyperparameter[self.dataset]['LLM_select_cells']
        )

        # 2. 混合检索补选 (利用刚才改进的 search_cell)
        # 注意：这里已经包含了 Retrieve + Rerank 逻辑
        retriever_cell_topk, retriever_match_id_list, _ = self.search_cell(
            query,
            topk=hyperparameter[self.dataset]['dense_topk'],
            LLM_match_id_list=LLM_match_id_list,
            select_cell=True # 告诉函数排除掉 LLM 已经选过的
        )

        # 3. 合并两部分结果
        # LLM 选的 + 检索器补的
        # 下面的代码负责把这些 ID 变成图结构 (Same Row / Same Column)
        # ... (保持原有的构建图代码逻辑不变) ...

        # 这种写法不仅代码更干净，而且利用了 Reranker 保证补选的质量极高。

        # 以下是原代码中构建返回值的逻辑，直接复用即可:
        cell_id_content_topk = OrderedDict({'LLM_select': {}, 'retriever_select': {}})
        cell_tuple_topk = copy.deepcopy(cell_id_content_topk)

        for i in range(len(LLM_cell_topk)):
            cell_id_content_topk['LLM_select'][LLM_match_id_list[i]] = LLM_cell_topk[i]
        cell_tuple_topk['LLM_select'] = self.cell2Tuple(LLM_match_id_list)

        for i in range(len(retriever_cell_topk)):
            cell_id_content_topk['retriever_select'][retriever_match_id_list[i]] = retriever_cell_topk[i]
        cell_tuple_topk['retriever_select'] = self.cell2Tuple(retriever_match_id_list)

        # 构建图连接
        # 合并所有ID用于构图
        all_ids = LLM_match_id_list + retriever_match_id_list
        hit_cell_same_row_col = self.getSameRow_ColCells(all_ids)
        connect_graph, connect_id_cell_dict, connect_graph_result = self.hit_cell_connect_graph(hit_cell_same_row_col)

        return cell_id_content_topk, cell_tuple_topk, connect_id_cell_dict, connect_graph_result

    def getSameRow_ColCells(self, match_cell_id):

        cols_num = len(self.cols)

        hit_cell_row_col_id = OrderedDict()
        for i in range(len(match_cell_id)):
            # result = topk_result[i]
            row_id = match_cell_id[i] // cols_num
            col_id = match_cell_id[i] % cols_num
            hit_cell_row_col_id[match_cell_id[i]] = {
                'row_id': [row_id],
                'col_id': [col_id]
            }

        hit_cell_same_row_col = OrderedDict()
        for key, value in hit_cell_row_col_id.items():
            row_id_list = value['row_id']
            col_id_list = value['col_id']
            hit_cell_same_row_col[key] = {
                'same_row_cells': [],
                'same_row_cells_id': [],
                'same_col_cells': [],
                'same_col_cells_id': []
            }
            for r_id in row_id_list:
                row_id_list = [i for i in range(r_id * len(self.rows[0]), (r_id + 1) * len(self.rows[0]))]
                hit_cell_same_row_col[key]['same_row_cells'].append(self.dealed_rows[r_id])
                hit_cell_same_row_col[key]['same_row_cells_id'].append(row_id_list)

            for c_id in col_id_list:
                col_is_list = [i for i in range(c_id, len(self.rows[0]) * (len(self.cols[c_id]) - 1) + c_id + 1,
                                                len(self.rows[0]))]
                hit_cell_same_row_col[key]['same_col_cells'].append(self.dealed_cols[c_id])
                hit_cell_same_row_col[key]['same_col_cells_id'].append(col_is_list)

        return hit_cell_same_row_col

    def hit_cell_connect_graph(self, order_info, get_shared_nei=False):

        # connect_graph = []
        connect_id_graph = []
        # order_info = hit_cell_same_row_col
        keys = list(order_info.keys())  # keys,_ = self.tableId2Content(keys)
        # 整理出同行同列
        for key, value in order_info.items():
            row_cells_id_list = value['same_row_cells_id']
            col_cells_id_list = value['same_col_cells_id']

            key_index = keys.index(key)
            # key_id = cell_id_list[key_index]
            for index in range(key_index + 1, len(keys)):
                n_key = keys[index]

                n_row_cells_id_list = order_info[n_key]['same_row_cells_id']
                n_col_cells_id_list = order_info[n_key]['same_col_cells_id']

                for row_cells_id in row_cells_id_list:
                    for i in range(len(row_cells_id)):
                        row_id = row_cells_id[i]
                        for n_row_cells_id in n_row_cells_id_list:
                            if row_cells_id == n_row_cells_id:
                                connect_id_graph.append([key, 'SAME ROW', n_key])
                            else:
                                for k in range(len(n_col_cells_id_list)):
                                    n_col_cells_id = n_col_cells_id_list[k]
                                    if row_id in n_col_cells_id:
                                        link_cell_id = n_col_cells_id.index(row_id)
                                        if key != n_col_cells_id_list[k][link_cell_id] and n_col_cells_id_list[k][
                                            link_cell_id] != n_key:
                                            connect_id_graph.append(
                                                [key, 'SAME ROW', n_col_cells_id_list[k][link_cell_id], 'SAME COLUMN',
                                                 n_key])

                for col_cells_id in col_cells_id_list:
                    for i in range(len(col_cells_id)):
                        col_id = col_cells_id[i]
                        for n_col_cells_id in n_col_cells_id_list:
                            if col_cells_id == n_col_cells_id:
                                connect_id_graph.append([key, 'SAME COLUMN', n_key])
                            else:
                                for k in range(len(n_row_cells_id_list)):
                                    n_row_cells_id = n_row_cells_id_list[k]
                                    if col_id in n_row_cells_id:
                                        link_cell_id = n_row_cells_id.index(col_id)
                                        if key != n_row_cells_id_list[k][link_cell_id] and n_row_cells_id_list[k][
                                            link_cell_id] != n_key:
                                            connect_id_graph.append(
                                                [key, 'SAME COLUMN', n_row_cells_id_list[k][link_cell_id], 'SAME ROW',
                                                 n_key])

        # 去重
        connect_id_graph = [tuple(i) for i in connect_id_graph if len(i) > 0]
        connect_id_graph = list(set(connect_id_graph))
        connect_id_graph = [list(i) for i in connect_id_graph]

        connect_graph_neighbors = OrderedDict()
        same_row_col_list = []
        row_width = len(self.rows[0])
        for i in range(len(connect_id_graph)):
            id_item = connect_id_graph[i]
            if len(id_item) > 3:
                pair_cell = [id_item[0], id_item[4]]
                if str(pair_cell) in connect_graph_neighbors.keys():
                    continue
                connect_graph_neighbors[str(pair_cell)] = [id_item[2]]
                for j in range(i + 1, len(connect_id_graph)):
                    next_id_item = connect_id_graph[j]
                    if len(next_id_item) > 3:
                        next_pair_cell = [next_id_item[0], next_id_item[4]]
                        if next_pair_cell == pair_cell:
                            connect_graph_neighbors[str(pair_cell)].append(next_id_item[2])
                connect_graph_neighbors[str(pair_cell)] = sorted(connect_graph_neighbors[str(pair_cell)])
            else:
                connect_graph_neighbors[str(id_item)] = id_item[1]
                same_row_col_list.append(id_item)

        connect_graph_tuples = OrderedDict()
        connect_graph = OrderedDict()
        connect_id_cell_dict = OrderedDict()
        for key, value in connect_graph_neighbors.items():
            pair_cell = ast.literal_eval(key)
            if len(pair_cell) == 2:
                temp = []
                tuple_temp = self.cell2Tuple(pair_cell)
                for cell in pair_cell:
                    row_id = cell // row_width
                    col_id = cell % row_width
                    connect_id_cell_dict[cell] = self.dealed_rows[row_id][col_id]
                    temp.append(self.dealed_rows[row_id][col_id])
                n_temp = []
                tuple_n_temp = self.cell2Tuple(value)
                for cell in value:
                    row_id = cell // row_width
                    col_id = cell % row_width
                    connect_id_cell_dict[cell] = self.dealed_rows[row_id][col_id]
                    n_temp.append(self.dealed_rows[row_id][col_id])
                connect_graph[str(temp)] = n_temp
                connect_graph_tuples[str(tuple_temp)] = tuple_n_temp

        connect_graph_result = []
        contents, _ = self.tableId2Content(keys, self.dealed_rows)
        for key_item in contents:  # 按检索的单元格顺序输出
            for key, value in connect_graph_tuples.items():
                temp_key = [key[2] for key in ast.literal_eval(key)]
                if temp_key[0] == key_item:
                    if type(value) == list:
                        connect_graph_result.append(
                            PROMPT_TEMPLATE[self.dataset]['shared_neighbors'].replace('{cell_pair}', key)
                            .replace(
                                '{shared_cells}', str(connect_graph_tuples[key])))

        same_row_col_result, same_row_col_tuple_result = self.organize_same_row_col(same_row_col_list, keys)
        if not get_shared_nei:
            connect_graph_result += same_row_col_tuple_result
            # connect_graph_result = same_row_col_tuple_result + connect_graph_result

        tuple_contents = self.cell2Tuple(keys)

        connect_graph_result = sorted(set(connect_graph_result), key=connect_graph_result.index)
        if len(connect_graph_result) == 0 and get_shared_nei:
            if len(same_row_col_result) > 0:
                if "SAME ROW" in same_row_col_result[0]:
                    connect_graph_result = PROMPT_TEMPLATE[self.dataset]['no_shared_neighbors_but_same_row'].replace(
                        '{cell_pair}', str(tuple_contents))
                else:
                    connect_graph_result = PROMPT_TEMPLATE[self.dataset]['no_shared_neighbors_but_same_col'].replace(
                        '{cell_pair}', str(tuple_contents))
            else:
                connect_graph_result = PROMPT_TEMPLATE[self.dataset]['no_shared_neighbors'].replace('{cell_pair}',
                                                                                                    str(tuple_contents))
        else:
            connect_graph_result = '\n'.join(connect_graph_result)

        return connect_graph, connect_id_cell_dict, connect_graph_result

    def tableId2Content(self, id_list, table, hit_cell_id=None):
        content_list = []
        content_id_list = []
        is_hit = False
        for id in id_list:
            row_id = id // len(table[0])
            col_id = id % len(table[0])
            content = table[row_id][col_id]
            if hit_cell_id != None:
                hit_cell_row_id = hit_cell_id // len(table[0])
                hit_cell_col_id = hit_cell_id % len(table[0])
                for rlo, rhi, clo, chi in self.merged_cells:
                    # 判断行下标是否在按列的合并单元格范围内
                    if row_id in range(rlo, rhi) and col_id in range(clo, chi) and hit_cell_row_id in range(rlo,
                                                                                                            rhi) and hit_cell_col_id in range(
                            clo, chi):
                        is_hit = True
                        break
            if is_hit:
                is_hit = False
                continue
            # if content:
            content_id_list.append(id)
            content_list.append(content)
        return content_list, content_id_list

    def LLM_select_cells_from_table(self, query, topk=3, prompt_tamplate='LLM_select_cells'):
        cells = table2Tuple(self.dealed_rows)
        if self.table_cation:
            table = f"Table Caption: {self.table_cation} \n**Table:**\n {str(cells)}"
        else:
            table = f"**Table:**\n{str(cells)}"

        system_instruction = PROMPT_TEMPLATE[self.dataset]['LLM_select_cells_system_instruction'].replace('{topk}',
                                                                                                          str(topk))
        examples = PROMPT_TEMPLATE[self.dataset]['LLM_select_cell_examples']

        prompt = PROMPT_TEMPLATE[self.dataset][prompt_tamplate] \
            .replace('{question}', query).replace('{table}', table).replace('{topk}', str(topk)).replace('{examples}',
                                                                                                         examples)

        if self.LLM_model.args.debug:
            print('大模型选择cell的prompt', prompt)

        # 修改此处的调用，添加 response_format 参数
        select_id = self.LLM_generate(
            prompt,
            system_instruction=system_instruction,
            response_format={"type": "json_object"}  # <-- 新增参数
        )

        print('模型初选的cell为', select_id)

        select_id_list = []
        for i in range(len(select_id)):
            if i >= topk:
                break
            cell_tuple_ = select_id[i]
            cell_tuple = cell_tuple_['tuple']
            cell_tuple = remove_quotes(cell_tuple).strip(',').strip('.').strip('，').strip('。').strip() if type(
                cell_tuple) == str else cell_tuple

            if type(cell_tuple) == str and cell_tuple.startswith('(') and cell_tuple.endswith(')'):
                try:
                    cell_tuple = ast.literal_eval(cell_tuple)
                except Exception as e:
                    print('LLM输出action无法解析', cell_tuple, e.__str__())
                    item_list = [i.strip().strip('(').strip(')').strip('"').strip("'").strip() for i in
                                 cell_tuple.split(',')]
                    cell_tuple = (int(item_list[0]), int(item_list[1]), ''.join(item_list[2:]))

            row_num = int(cell_tuple[0])
            col_num = int(cell_tuple[1])
            cell_id = len(self.dealed_rows[0]) * row_num + col_num

            if cell_id in self.cell_ids:
                select_id_list.append(cell_id)

        result, _ = self.tableId2Content(select_id_list, self.dealed_rows)

        return result, select_id_list

    def check_arg_exists(self, arg):
        if arg in self.cells:
            cell_id = self.cells.index(arg)
            return cell_id, True
        else:
            return None, False

    def organize_same_row_col(self, same_row_col_list, keys):
        link_same_row_col = []
        for i in range(len(same_row_col_list)):
            item = copy.deepcopy(same_row_col_list[i])
            if len(item) == 0:
                continue
            if item[1] == 'SAME ROW':
                id = item[0] // len(self.rows[0])
            else:
                id = item[0] % len(self.rows[0])

            for j in range(i + 1, len(same_row_col_list)):
                n_item = same_row_col_list[j]
                if len(n_item) == 0:
                    continue
                if item[1] == 'SAME ROW':
                    n_id = n_item[0] // len(self.rows[0])
                else:
                    n_id = n_item[0] % len(self.rows[0])

                if n_item[1] == item[1] and n_id == id:
                    if n_item[0] in item:
                        item.append(n_item[1])
                        item.append(n_item[2])
                    elif n_item[2] in item:
                        item.append(n_item[1])
                        item.append(n_item[0])
                    else:
                        item += [n_item[1]] + n_item
                    same_row_col_list[j] = []
            link_same_row_col.append(item)

        result = []
        tuple_result = []
        for key_item in keys:
            for i in range(len(link_same_row_col)):
                item = link_same_row_col[i]
                id_list = sorted(list(set([k for k in item if type(k) == int])))
                if key_item == id_list[0]:
                    temp = []
                    tuple_item = self.cell2Tuple(id_list)
                    for j in range(len(id_list)):
                        cell = id_list[j]
                        row_id = cell // len(self.dealed_rows[0])
                        col_id = cell % len(self.dealed_rows[0])
                        temp.append("'{}'".format(self.dealed_rows[row_id][col_id]))
                        # tuple_item.append((row_id,col_id,"{}".format(self.dealed_rows[row_id][col_id])))
                    temp = '[{}]'.format(',{},'.format(item[1]).join(temp))
                    tuple_item = '[{}]'.format(',{},'.format(item[1]).join([str(k) for k in tuple_item]))
                    result.append(temp)
                    tuple_result.append(tuple_item)
        return result, tuple_result

    def cell2Tuple(self, id_list, add_id=False, add_merged_cells=False):
        tuple_result = []
        merged_id_list = []
        merged_cotent_list = []
        for i in range(len(id_list)):
            cell_id = id_list[i]
            row_id = cell_id // len(self.rows[0])
            col_id = cell_id % len(self.rows[0])
            content = copy.deepcopy(self.dealed_rows[row_id][col_id])
            content = str(content).replace('"', "'")

            if add_id:
                temp = (cell_id, row_id, col_id, content)
            else:
                temp = (row_id, col_id, content)
            if temp not in tuple_result:
                merged_cotent_list.append(content)
                merged_id_list.append(cell_id)
                tuple_result.append(temp)

            if add_merged_cells:
                for rlo, rhi, clo, chi in self.merged_cells:
                    if row_id in range(rlo, rhi) and col_id in range(clo, chi):
                        for r in range(rlo, rhi):
                            for c in range(clo, chi):
                                if (r, c, content) not in tuple_result:
                                    merged_cotent_list.append(self.dealed_rows[r][c])
                                    merged_id_list.append(r * len(self.rows[0]) + c)
                                    tuple_result.append((r, c, content))
                        break
        if add_merged_cells:
            return merged_cotent_list, merged_id_list, tuple_result

        return tuple_result
