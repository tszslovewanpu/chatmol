import matplotlib.pyplot as plt
import selfies as sf
import pandas as pd
import numpy as np
import json
import time
import csv
import os
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from multiprocessing import Pool
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem.QED import qed
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from collections import Counter
import statistics
import jsonlines

class DataProcessor:
    def __init__(self):
        self.total_inference = []
        self.simplified_with_three_quotes = []
        self.simplified_with_newline = []
        self.selfies_unbalanced = []
        self.selfies2smiles_failed = []
        self.smiles_unvalid = []
        self.fail2compute_property = []
        self.success_property = []

    def function_A(self, simplified, simplified_selfies_smiles_logp_csv_file, prompt):
        item = simplified ## item就是生成的simplified string
        self.total_inference.append(item)

        if item.count("'") >= 3:
            self.simplified_with_three_quotes.append(item)
            print(f"Error item '{item}' appended to simplified_with_three_quotes list")
            return np.nan

        if '\n' in item:
            self.simplified_with_newline.append(item)
            print(f"Error item '{item}' appended to simplified_with_newline list")
            return np.nan

        selfies_string = '[' + item.strip("'").replace(' ', '][') + ']'
        balance = 0
        for char in selfies_string:
            if char == '[':
                balance += 1
            elif char == ']':
                balance -= 1

        if balance < 0:
            self.selfies_unbalanced.append(item)
            print(f"Error item '{item}' appended to selfies_unbalanced list")
            return np.nan
        elif balance == 0:
            try:
                smiles_string = sf.decoder(selfies_string)
            except Exception as e:
                self.selfies2smiles_failed.append(item)
                print(f"Error item '{item}' appended to selfies2smiles_failed list")
                print(f"Error occurred while decoding selfies: {e}")
                return np.nan
            try:
                can_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_string), isomericSmiles=True)
                mol = Chem.MolFromSmiles(can_smiles)
                AllChem.Compute2DCoords(mol)
            except Exception as e:
                self.smiles_unvalid.append(item)
                print(f"Error item '{item}' appended to smiles_unvalid list")
                print(f"Error occurred while computing 2D coordinates: {e}")
                return np.nan
            try:
                mol = Chem.MolFromSmiles(smiles_string)
                # logp = Descriptors.MolLogP(mol)
                sa = sascorer.calculateScore(mol)
                mol_qed = qed(mol)
            except Exception as e:
                self.fail2compute_property.append(item)
                print(f"Error item '{item}' appended to fail2compute_property list")
                print(f"Error occurred while computing molecular qed or sa: {e}")
                return np.nan
            # Incremental writing to CSV file
            with open(simplified_selfies_smiles_logp_csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([prompt, item, selfies_string, smiles_string, sa, mol_qed])

            self.success_property.append(item)
            return 1

        return np.nan

    def process_jsonl(self, input_file, simplified_selfies_smiles_logp_csv_file):
        # with open(input_file, 'r') as f, \
        #         open(output_success_file, 'w') as f_success, \
        #         open(output_fail_file, 'w') as f_fail:
        with open(input_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                simplified = data.get("Output")
                prompt = data.get("Input")
                if simplified is not None:
                    # logp = self.function_A(simplified, simplified_selfies_smiles_logp_csv_file)
                    self.function_A(simplified, simplified_selfies_smiles_logp_csv_file, prompt)
                    # if np.isnan(logp):
                    #     data["logp"] = None  # If logp is np.nan, set logp to None
                    #     json.dump(data, f_fail)
                    #     f_fail.write('\n')
                    # else:
                    #     data["logp"] = logp
                    #     json.dump(data, f_success)
                    #     f_success.write('\n')

        print("total_inference:", len(self.total_inference))
        print("simplified_with_three_quotes:", len(self.simplified_with_three_quotes))
        print("simplified_with_newline:", len(self.simplified_with_newline))
        print("selfies_unbalanced:", len(self.selfies_unbalanced))
        print("selfies2smiles_failed:", len(self.selfies2smiles_failed))
        print("smiles_unvalid:", len(self.smiles_unvalid))
        print("fail2compute_property:", len(self.fail2compute_property))
        print("success_compute property:", len(self.success_property))


    def logp_in_range(self):
        logp = self.df['logp']
        count_in_range = logp[(logp > -2.5) & (logp < -2)].count()
        total_count = logp.count()
        percentage = (count_in_range / total_count) * 100
        print("区间范围(-2.5, -2)开区间内的分子个数:", count_in_range)
        print("所占百分比:", percentage, "%\n")

    # def similarity_calculation(self, idx1, df):
    def similarity_calculation(self, idx1):
        fp1 = self.df.at[idx1, 'mfp2']
        simlist = [DataStructs.DiceSimilarity(fp1, self.df.at[idx2, 'mfp2']) for idx2 in self.df.index]
        return sum(simlist)

    # def similarity_calculation_wrapper(self, idx1):
    #     # return self.similarity_calculation(idx1, self.df) ## 自己的self.df不用传的
    #     return self.similarity_calculation(idx1)

    def compute_similarity(self, simplified_selfies_smiles_logp_csv_file):
        df1 = pd.read_csv(simplified_selfies_smiles_logp_csv_file)
        # 在计算之前，检查smiles列中的数据类型
        # 保留smiles列中不是float类型的数据
        df2 = df1[~df1['smiles'].apply(lambda x: isinstance(x, float))]
        self.df = df2
        # self.logp_in_range(self.df)
        # self.logp_in_range()

        start_time1 = time.time()
        PandasTools.AddMoleculeColumnToFrame(self.df, 'smiles', 'mol', includeFingerprints=True)
        end_time1 = time.time()
        print("Time taken to add molecules to dataframe:", end_time1 - start_time1, "seconds")

        start_time2 = time.time()
        fplist = [Chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in self.df['mol']]  ## 2的意思是r = 2 半径
        self.df['mfp2'] = fplist
        end_time2 = time.time()
        print("Time taken to compute Morgan fingerprints and add to dataframe:", end_time2 - start_time2, "seconds")

        start_time3 = time.time()
        with Pool() as pool:
            # similarity_sum = sum(pool.map(self.similarity_calculation_wrapper, df.index)) ## map（函数，可迭代器）
            similarity_sum = sum(pool.map(self.similarity_calculation, self.df.index)) ## map（函数，可迭代器）

        print(f"Number of molecules: {df1.shape[0]}")
        print(f"Number of molecules that can be converted into mol: {df2.shape[0]}")
        print(f"Similarity summarization with comparison with self comparision is: {similarity_sum}")
        print(f"The number of per-pair without self comparison is: {((self.df.shape[0] ** 2) - self.df.shape[0])}")
        print(f"Similarity summarization without self comparison is: {similarity_sum - self.df.shape[0]}")
        similarity_average = (similarity_sum - self.df.shape[0]) / ((self.df.shape[0] ** 2) - self.df.shape[0])
        print(f"The average similarity is: {similarity_average}")
        end_time3 = time.time()
        print(f"Time taken to compute similarity: {end_time3 - start_time3}\n")

    def duplicate_trainingset(self, jsonl_dir, csv_file):
        data_set_training = set()
        with open(jsonl_dir, 'r', encoding='utf-8') as file:
            for line in file:
                json_data = json.loads(line.strip())
                conversation = json_data['conversations']
                for item in conversation:
                    if item['from'] == 'gpt':
                        data_set_training.add(item['value'].strip("'"))
        data_set_generated = set()
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data_set_generated.add(row['simplified'].strip("'"))
        
        # compute the duplicated data
        duplicate_data = data_set_generated.intersection(data_set_training)
        duplicate_rate = len(duplicate_data) / len(data_set_generated) * 100

        print("number of duplicated data: ", len(duplicate_data))
        print("rate of duplicated data with trainingset : {:.2f}%".format(duplicate_rate))

    def draw_distribution(self, values, property_name, csv_file):
        value_counts = values.value_counts()
        plt.bar(value_counts.index, value_counts.values)
        plt.title('Discrete Value Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        # 构建保存图片的路径
        save_path = os.path.join(os.path.dirname(csv_file), f'{property_name}_discrete_value_distribution.png')
        plt.savefig(save_path)
        # plt.show()
    
    def analysis(self, property_name, csv_file):
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        # 提取表头A的数据
        column_a_data = df[f'{property_name}']
        # 计算最大值、最小值、平均值、方差、中位数
        max_value = column_a_data.max()
        min_value = column_a_data.min()
        mean_value = column_a_data.mean()
        variance_value = column_a_data.var()
        median_value = column_a_data.median()
        # 计算众数
        mode_value = column_a_data.mode()[0]
        # 打印结果
        print(f"analyze {property_name}")
        print("最大值:", max_value)
        print("最小值:", min_value)
        print("平均值:", mean_value)
        print("方差:", variance_value)
        print("中位数:", median_value)
        print("众数:", mode_value)
        print("-----------------\n")
    
    def filter_inference_result_with_training_set(self, training_jsonl_file, inference_output_jsonl_file, novel_inference_output_jsonl_file):
        # 读取jsonl1文件，提取conversation键的值，然后取其[1]的部分，然后取其value的值
        with jsonlines.open(training_jsonl_file) as reader:
            jsonl1_data = [line['conversations'][1]['value'] for line in reader]

        # 读取jsonl2文件，取其Output的值
        with jsonlines.open(inference_output_jsonl_file) as reader:
            jsonl2_data = [{'Input': line['Input'], 'Output': line['Output']} for line in reader]
        # 初始化重复率计数器
        num_duplicates = 0
        # 将jsonl2中与jsonl1重复的数据剔除，保存为new_jsonl2文件
        filtered_jsonl2_data = []
        for line in jsonl2_data:
            if line not in jsonl1_data:
                filtered_jsonl2_data.append(line)
            else:
                num_duplicates += 1

        # 计算重复率
        total_lines = len(jsonl2_data)
        duplicate_rate = num_duplicates / total_lines

        # 输出重复率
        print("重复率: {:.2%}".format(duplicate_rate))
        # 将剔除重复数据后的jsonl2数据保存为new_jsonl2文件
        with jsonlines.open(novel_inference_output_jsonl_file, mode='w') as writer:
            for line in filtered_jsonl2_data:
                writer.write({'Input': line['Input'], 'Output': line['Output']})
                
        # # 将jsonl2中与jsonl1重复的数据剔除，保存为new_jsonl2文件
        # filtered_jsonl2_data = [line for line in jsonl2_data if line not in jsonl1_data]

        # with jsonlines.open(novel_inference_output_jsonl_file, mode='w') as writer:
        #     for line in filtered_jsonl2_data:
        #         writer.write({'Output': line})
    
def main():
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取上一级目录
    inference_process_path = os.path.dirname(current_dir)
    # inference_process_path = '/data1/fcl/workspace/2024_71/240610_35_xiaorongshiyan/multi_properties_llama3/inference/inference_1400_16'
    inference_output_process_path = f'{inference_process_path}/inference_output_process'
    training_jsonl_file = '/data/fcl/fcl/workspace/2024_35/240528_35_xiaorongshiyan/llama3/numerical_enhancement/multi/1err/dataset/sft_qed_sa_kd_no_norm_decimal2_kd_small.jsonl'

    # Instantiate the DataProcessor class
    processor = DataProcessor()
    # Define input and output file names
    inference_output_jsonl_file = f'{inference_process_path}/result/prediction_result.jsonl'
    # filter and save file name
    novel_inference_output_jsonl_file = f'{inference_output_process_path}/novel_prediction_result.jsonl'
    
    # 与训练集去重
    processor.filter_inference_result_with_training_set(training_jsonl_file, inference_output_jsonl_file, novel_inference_output_jsonl_file)

    # 等会结果保存在这里
    valid_novel_simplified_selfies_smiles_qed_sa_csv_file = f'{inference_output_process_path}/valid_novel_simplified_selfies_smiles_qed_sa.csv'
    with open(valid_novel_simplified_selfies_smiles_qed_sa_csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['simplified', 'selfies', 'smiles', 'sa', 'qed'])
        writer.writerow(['prompt', 'simplified', 'selfies', 'smiles', 'sa', 'qed'])


    # Process JSONL file    # 已经处理好了，不能一次执行一条，因为这样会导致csv重新加载，变空
    processor.process_jsonl(novel_inference_output_jsonl_file, valid_novel_simplified_selfies_smiles_qed_sa_csv_file)
    # Process csv file#
    # processor.compute_similarity(simplified_selfies_smiles_logp_csv_file)
    # processor.duplicate_trainingset(training_jsonl_file, simplified_selfies_smiles_qed_sa_csv_file)
    
    # # draw distribution
    # data = pd.read_csv(valid_novel_simplified_selfies_smiles_qed_sa_csv_file)
    # processor.draw_distribution(data['sa'], 'sa', valid_novel_simplified_selfies_smiles_qed_sa_csv_file)
    # processor.draw_distribution(data['qed'], 'qed', valid_novel_simplified_selfies_smiles_qed_sa_csv_file)

    processor.analysis('sa', valid_novel_simplified_selfies_smiles_qed_sa_csv_file)
    processor.analysis('qed', valid_novel_simplified_selfies_smiles_qed_sa_csv_file)

    # processor = DataProcessor()
    # processor.analysis('1err_kd', '/data1/fcl/workspace/2024_71/240505_71_multi/inference/inference_output_process/qed_sa_1err.csv')

if __name__ == "__main__":
    main()
# (matllm) fcl@amax:~$ python /data1/fcl/workspace/2024_73/240326_71_round2_scratch_73_inference/inference_epoch60_240404/inference_output_process/analysis.py > /data1/fcl/workspace/2024_73/240326_71_round2_scratch_73_inference/inference_epoch60_240404/inference_output_process/log240407.log 

