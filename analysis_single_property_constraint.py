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
import seaborn as sns
import argparse
from collections import Counter

class DataProcessor:
    def __init__(self):
        self.novel_inference = []
        self.simplified_with_three_quotes = []
        self.simplified_with_newline = []
        self.selfies_unbalanced = []
        self.selfies2smiles_failed = []
        self.smiles_unvalid = []
        self.fail2compute_property = []
        self.success_logp = []

    def function_A(self, simplified, simplified_selfies_smiles_logp_csv_file):
        item = simplified ## item就是生成的simplified string
        self.novel_inference.append(item)

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
                logp = Descriptors.MolLogP(mol)
            except Exception as e:
                self.fail2compute_property.append(item)
                print(f"Error item '{item}' appended to fail2compute_property list")
                print(f"Error occurred while computing molecular logP: {e}")
                return np.nan
            # Incremental writing to CSV file
            with open(simplified_selfies_smiles_logp_csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([item, selfies_string, smiles_string, logp])

            self.success_logp.append(logp)
            return logp

        return np.nan
    
    def filter_and_count_logp(self, summary_csv_file):
        filtered_data = []
        
        # Read and filter the CSV file
        with open(summary_csv_file, 'r') as f_csv:
            csv_reader = csv.reader(f_csv)
            header = next(csv_reader)  # Skip header
            
            for row in csv_reader:
                logp = float(row[2]) if row[2] else None
                if logp is not None and -2.5 <= logp and logp <= -2:
                    filtered_data.append(row)
        
        # Count frequencies of 'Input'
        input_counter = Counter(row[0] for row in filtered_data)
        
        # Print the frequency counts
        for input_value, count in input_counter.items():
            print(f"{input_value}: {count}")

    def process_jsonl(self, input_file, output_success_file, output_fail_file, simplified_selfies_smiles_logp_csv_file, summary_csv_file):
        with open(input_file, 'r') as f, \
                open(output_success_file, 'w') as f_success, \
                open(output_fail_file, 'w') as f_fail, \
                open(summary_csv_file, 'w', newline='') as f_csv:
            
            csv_writer = csv.writer(f_csv)
            csv_writer.writerow(["Input", "Output", "logp"])  # Write header
        
            for line in f:
                data = json.loads(line.strip())
                simplified = data.get("Output")
                if simplified is not None:
                    logp = self.function_A(simplified, simplified_selfies_smiles_logp_csv_file)
                    if np.isnan(logp):
                        data["logp"] = None  # If logp is np.nan, set logp to None
                        json.dump(data, f_fail)
                        f_fail.write('\n')
                    else:
                        data["logp"] = logp
                        json.dump(data, f_success)
                        f_success.write('\n')
                    # Add to CSV file
                    csv_writer.writerow([data.get("Input"), simplified, logp])

        print("新分子个数:", len(self.novel_inference))
        print("------------------------------------")
        print("非法类型统计：")
        print("simplified_with_three_quotes:", len(self.simplified_with_three_quotes))
        print("simplified_with_newline:", len(self.simplified_with_newline))
        print("selfies_unbalanced:", len(self.selfies_unbalanced))
        print("selfies2smiles_failed:", len(self.selfies2smiles_failed))
        print("smiles_unvalid:", len(self.smiles_unvalid))
        print("fail2compute_property:", len(self.fail2compute_property))
        print("------------------------------------")
        print(f"非法分子个数：{len(self.novel_inference) - len(self.success_logp)}")
        print("合法分子个数:", len(self.success_logp))
        print("--------------------------------")

    # def logp_in_range(self):
    #     logp = self.df['logp']
    #     count_in_range = logp[(logp >= -2.5) & (logp <= -2)].count()
    #     total_count = logp.count()
    #     percentage = (count_in_range / total_count) * 100
    #     print("区间范围[-2.5, -2]闭区间内的分子个数:", count_in_range)
    #     print("所占百分比:", percentage, "%\n")
    # def logp_in_range(self):
    #     logp = self.df['logp'].round(1)  # 四舍五入到一位小数
    #     count_in_range = logp[(logp >= -2.5) & (logp <= -2.0)].count()
    #     total_count = logp.count()
    #     percentage = (count_in_range / total_count) * 100
    #     print("区间范围[-2.5, -2]闭区间内的分子个数:", count_in_range)
    #     print("所占百分比:", percentage, "%\n")


    def logp_in_range(self, logp_inrange_csv):
        logp = self.df['logp'].round(1)  # 四舍五入到一位小数
        count_in_range = logp[(logp >= -2.5) & (logp <= -2.0)].count()
        total_count = logp.count()
        percentage = (count_in_range / total_count) * 100
        # percentage = (count_in_range / len(self.novel_inference)) * 100

        print("符合条件分子个数:", count_in_range)
        print("符合条件分子占合法分子的比例:", percentage, "%\n")

        # 筛选符合范围的行
        self.df_in_range = self.df[(logp >= -2.5) & (logp <= -2.0)]

        # 保存为csv文件
        self.df_in_range.to_csv(logp_inrange_csv, index=False)


    # # def similarity_calculation(self, idx1, df):
    # def similarity_calculation(self, idx1):
    #     fp1 = self.df.at[idx1, 'mfp2']
    #     simlist = [DataStructs.DiceSimilarity(fp1, self.df.at[idx2, 'mfp2']) for idx2 in self.df.index]
    #     return sum(simlist)
    # def similarity_calculation(self, idx1, df):
    def similarity_calculation(self, idx1):
        fp1 = self.df_in_range.at[idx1, 'mfp2']
        simlist = [DataStructs.DiceSimilarity(fp1, self.df_in_range.at[idx2, 'mfp2']) for idx2 in self.df_in_range.index]
        return sum(simlist)

    # def similarity_calculation_wrapper(self, idx1):
    #     # return self.similarity_calculation(idx1, self.df) ## 自己的self.df不用传的
    #     return self.similarity_calculation(idx1)

    # def compute_similarity(self, simplified_selfies_smiles_logp_csv_file):
    #     df1 = pd.read_csv(simplified_selfies_smiles_logp_csv_file)
    #     # 在计算之前，检查smiles列中的数据类型
    #     # 保留smiles列中不是float类型的数据
    #     df2 = df1[~df1['smiles'].apply(lambda x: isinstance(x, float))]
    #     self.df = df2
    #     # self.logp_in_range(self.df)
    #     self.logp_in_range()

    #     start_time1 = time.time()
    #     PandasTools.AddMoleculeColumnToFrame(self.df, 'smiles', 'mol', includeFingerprints=True)
    #     end_time1 = time.time()
    #     print("Time taken to add molecules to dataframe:", end_time1 - start_time1, "seconds")

    #     start_time2 = time.time()
    #     fplist = [Chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in self.df['mol']]  ## 2的意思是r = 2 半径
    #     self.df['mfp2'] = fplist
    #     end_time2 = time.time()
    #     print("Time taken to compute Morgan fingerprints and add to dataframe:", end_time2 - start_time2, "seconds")

    #     start_time3 = time.time()
    #     with Pool() as pool:
    #         # similarity_sum = sum(pool.map(self.similarity_calculation_wrapper, df.index)) ## map（函数，可迭代器）
    #         similarity_sum = sum(pool.map(self.similarity_calculation, self.df.index)) ## map（函数，可迭代器）

    #     print(f"Number of molecules: {df1.shape[0]}")
    #     print(f"Number of molecules that can be converted into mol: {df2.shape[0]}")
    #     print(f"Similarity summarization with comparison with self comparision is: {similarity_sum}")
    #     print(f"The number of per-pair without self comparison is: {((self.df.shape[0] ** 2) - self.df.shape[0])}")
    #     print(f"Similarity summarization without self comparison is: {similarity_sum - self.df.shape[0]}")
    #     similarity_average = (similarity_sum - self.df.shape[0]) / ((self.df.shape[0] ** 2) - self.df.shape[0])
    #     print(f"The average similarity is: {similarity_average}")
    #     end_time3 = time.time()
    #     print(f"Time taken to compute similarity: {end_time3 - start_time3}\n")


    def compute_similarity(self, simplified_selfies_smiles_logp_csv_file, logp_inrange_csv):
        df1 = pd.read_csv(simplified_selfies_smiles_logp_csv_file)
        # 在计算之前，检查smiles列中的数据类型
        # 保留smiles列中不是float类型的数据
        df2 = df1[~df1['smiles'].apply(lambda x: isinstance(x, float))]
        self.df = df2
        # self.logp_in_range(self.df)
        self.logp_in_range(logp_inrange_csv)

        start_time1 = time.time()
        PandasTools.AddMoleculeColumnToFrame(self.df_in_range, 'smiles', 'mol', includeFingerprints=True)
        end_time1 = time.time()
        # print("Time taken to add molecules to dataframe:", end_time1 - start_time1, "seconds")

        start_time2 = time.time()
        fplist = [Chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in self.df_in_range['mol']]  ## 2的意思是r = 2 半径
        self.df_in_range['mfp2'] = fplist
        end_time2 = time.time()
        # print("Time taken to compute Morgan fingerprints and add to dataframe:", end_time2 - start_time2, "seconds")

        start_time3 = time.time()
        with Pool() as pool:
            # similarity_sum = sum(pool.map(self.similarity_calculation_wrapper, df.index)) ## map（函数，可迭代器）
            similarity_sum = sum(pool.map(self.similarity_calculation, self.df_in_range.index)) ## map（函数，可迭代器）

        # print(f"Number of molecules: {df1.shape[0]}")
        # print(f"Number of molecules that can be converted into mol: {df2.shape[0]}")
        # print(f"Similarity summarization with comparison with self comparision is: {similarity_sum}")
        # print(f"The number of per-pair without self comparison is: {((self.df.shape[0] ** 2) - self.df.shape[0])}")
        # print(f"Similarity summarization without self comparison is: {similarity_sum - self.df_in_range.shape[0]}")
        similarity_average = (similarity_sum - self.df_in_range.shape[0]) / ((self.df_in_range.shape[0] ** 2) - self.df_in_range.shape[0])
        print("--------------------------------")
        print(f"符合条件分子相似度为：{similarity_average}")
        print(f"符合条件分子多样性为: {1.0 - similarity_average}")
        end_time3 = time.time()
        # print(f"Time taken to compute similarity: {end_time3 - start_time3}\n")

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

    def draw_distribution(self, csv_file):
        data = pd.read_csv(csv_file)
        values = data.iloc[:, -1]
        value_counts = values.value_counts()
        plt.bar(value_counts.index, value_counts.values)
        plt.title('Discrete Value Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        # 构建保存图片的路径
        save_path = os.path.join(os.path.dirname(csv_file), 'discrete_value_distribution.png')
        plt.savefig(save_path)
        # plt.show()

    def draw(self, file_paths, save_path):
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            if "zinc_250k_properties_chemical_and_protein.csv" in file_path:
                label = "Trainingset"
            elif "simplified_selfies_smiles_logp.csv" in file_path:
                label = "LlaMol"
            else:
                label = None
            sns.kdeplot(data=df, x="logp", label=label)  # Plot KDE plot for each file
        plt.xlabel("logp")
        plt.ylabel("Density")
        plt.title("Distribution of logp")
        plt.legend()
        plt.savefig(f'{save_path}/distribution_logp_comparision.png')
        plt.show()

# def quchong(training_jsonl_file, inference_file, noval_inference_file):
#     import jsonlines

#     # 读取trainingset文件，提取conversation键的值，然后取其[1]的部分，然后取其value的值
#     with jsonlines.open(training_jsonl_file) as reader:
#         jsonl1_data = [line['conversations'][1]['value'] for line in reader]

#     # 读取inference文件，取其Output的值
#     with jsonlines.open(inference_file) as reader:
#         jsonl2_data = [line['Output'] for line in reader]

#     # 将jsonl2中与jsonl1重复的数据剔除，保存为new_jsonl2文件
#     filtered_jsonl2_data = [line for line in jsonl2_data if line not in jsonl1_data]
    
#     total_lines_in_inference_file = len(jsonl2_data)
#     total_unique_lines_in_inference_file = len(filtered_jsonl2_data)
#     duplicate_rate = (total_lines_in_inference_file - total_unique_lines_in_inference_file) / total_lines_in_inference_file
#     print("Duplicate rate with trainingset:", duplicate_rate)

#     with jsonlines.open(noval_inference_file, mode='w') as writer:
#         for line in filtered_jsonl2_data:
#             writer.write({'Output': line})
def quchong(training_jsonl_file, inference_file, noval_inference_file):
    import jsonlines

    # 读取trainingset文件，提取conversation键的值，然后取其[1]的部分，然后取其value的值
    with jsonlines.open(training_jsonl_file) as reader:
        jsonl1_data = [line['conversations'][1]['value'] for line in reader]

    # 读取inference文件，取其每行的Input和Output键值对
    with jsonlines.open(inference_file) as reader:
        jsonl2_data = [{'Input': line['Input'], 'Output': line['Output']} for line in reader]
    line_count = len(jsonl2_data)
    print("--------------------------------")
    print(f"总计推理的分子个数: {line_count}")
    print("--------------------------------")
    # 将jsonl2中与jsonl1重复的Output数据剔除，保留剩余的数据
    filtered_jsonl2_data = [line for line in jsonl2_data if line['Output'] not in jsonl1_data]
    
    total_lines_in_inference_file = len(jsonl2_data)
    total_unique_lines_in_inference_file = len(filtered_jsonl2_data)
    duplicate_rate = (total_lines_in_inference_file - total_unique_lines_in_inference_file) / total_lines_in_inference_file
    print(f"与训练集重复分子个数：{(total_lines_in_inference_file - total_unique_lines_in_inference_file)}")
    print("与训练集重复率:", duplicate_rate)

    # 将过滤后的数据写入noval_inference_file文件
    with jsonlines.open(noval_inference_file, mode='w') as writer:
        for line in filtered_jsonl2_data:
            writer.write({'Input': line['Input'], 'Output': line['Output']})

# 示例调用
# quchong('trainingset.jsonl', 'inference.jsonl', 'noval_inference.jsonl')

# def main():
#     base_workspace = '/data/fcl/fcl/workspace/2024_35/240528_35_xiaorongshiyan/llama3/logp_targetting_-2/simplified/inference/inference_process'
#     training_jsonl_file = '/data/fcl/fcl/workspace/2024_35/240528_35_xiaorongshiyan/llama3/logp_targetting_-2/simplified/sft_training/dataset/zinc15_zinc250k_logp_inrange_round2.jsonl'
#     zinc250k_properties_csv_path = '/data/fcl/fcl/workspace/2024_35/240528_35_xiaorongshiyan/llama3/logp_targetting_-2/simplified/sft_training/dataset_csv/zinc_250k_properties_chemical_and_protein.csv'
    
#     output_save_path = f'{base_workspace}/inference_output_process'
#     inference_file = f'{base_workspace}/prediction_result.jsonl'
#     noval_inference_file = f'{output_save_path}/novel_inference_output.jsonl'
#     output_success_file = f'{output_save_path}/successfully_computed.jsonl'
#     output_fail_file = f'{output_save_path}/fail_to_compute.jsonl'
#     simplified_selfies_smiles_logp_csv_file = f'{output_save_path}/valid_simplified_selfies_smiles_logp.csv'

#     quchong(training_jsonl_file, inference_file, noval_inference_file)

#     with open(simplified_selfies_smiles_logp_csv_file, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['simplified', 'selfies', 'smiles', 'logp'])

#     processor = DataProcessor()
#     processor.process_jsonl(noval_inference_file, output_success_file, output_fail_file, simplified_selfies_smiles_logp_csv_file)
    
#     paths_list = [zinc250k_properties_csv_path, simplified_selfies_smiles_logp_csv_file]
#     processor.draw(paths_list, output_save_path)
#     processor.compute_similarity(simplified_selfies_smiles_logp_csv_file)

# if __name__ == "__main__":
#     main()






def main():
    parser = argparse.ArgumentParser(description='Inference process script')
    parser.add_argument('--base_workspace', type=str, required=True, help='Base workspace directory')
    parser.add_argument('--training_jsonl_file', type=str, required=True, help='Path to training JSONL file')
    # parser.add_argument('--zinc250k_properties_csv_path', type=str, required=True, help='Path to zinc250k properties CSV file')

    args = parser.parse_args()

    base_workspace = args.base_workspace
    training_jsonl_file = args.training_jsonl_file
    # zinc250k_properties_csv_path = args.zinc250k_properties_csv_path
    
    output_save_path = f'{base_workspace}/process_mine'
    inference_file = f'{base_workspace}/prediction_result.jsonl'
    noval_inference_file = f'{output_save_path}/novel_inference_output.jsonl'
    output_success_file = f'{output_save_path}/successfully_computed.jsonl'
    output_fail_file = f'{output_save_path}/fail_to_compute.jsonl'
    simplified_selfies_smiles_logp_csv_file = f'{output_save_path}/valid_simplified_selfies_smiles_logp.csv'
    logp_inrange_csv = f'{output_save_path}/logp_inrange_file.csv'
    in_out_logp_csv = f'{output_save_path}/in_out_logp.csv'

    quchong(training_jsonl_file, inference_file, noval_inference_file)

    with open(simplified_selfies_smiles_logp_csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['simplified', 'selfies', 'smiles', 'logp'])

    processor = DataProcessor()
    processor.process_jsonl(noval_inference_file, output_success_file, output_fail_file, simplified_selfies_smiles_logp_csv_file, in_out_logp_csv)
    
    # paths_list = [zinc250k_properties_csv_path, simplified_selfies_smiles_logp_csv_file]
    # processor.draw(paths_list, output_save_path)
    processor.compute_similarity(simplified_selfies_smiles_logp_csv_file, logp_inrange_csv)

    processor.filter_and_count_logp(in_out_logp_csv)

if __name__ == "__main__":
    main()
