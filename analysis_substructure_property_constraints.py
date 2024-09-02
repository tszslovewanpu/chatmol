import matplotlib.pyplot as plt
import selfies as sf
import pandas as pd
import numpy as np
import json
import time
import csv
import os
import re
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from multiprocessing import Pool
from rdkit.Chem import AllChem
from rdkit import DataStructs
import seaborn as sns
import argparse
import pandas as pd
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem.QED import qed
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcTPSA
import argparse
import json
import csv
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import logging
from datetime import datetime

class DataProcessor:
    def __init__(self):
        self.novol_inference = []
        self.simplified_with_three_quotes = []
        self.simplified_with_newline = []
        self.selfies_unbalanced = []
        self.selfies2smiles_failed = []
        self.smiles_unvalid = []
        self.fail2compute_property = []
        self.contain_substructure = []
        self.dont_contain_substructure = []
        self.fail2processHasSubstructMatch = []
        self.continuous_carbon = []

    # def get_largest_ring_size(mol):
    #     cycle_list = mol.GetRingInfo().AtomRings()
    #     if cycle_list:
    #         cycle_length = max(len(j) for j in cycle_list)
    #     else:
    #         cycle_length = 0
    #     return cycle_length

    def function_A(self, simplified, fragment, simplified_selfies_smiles_logp_csv_file, prompt):
        item = simplified ## item就是生成的simplified string
        self.novol_inference.append(item)

        if item.count("'") >= 3:
            self.simplified_with_three_quotes.append(item)
            print(f"Error item '{item}' appended to simplified_with_three_quotes list")
            return np.nan

        if '\n' in item:
            self.simplified_with_newline.append(item)
            print(f"Error item '{item}' appended to simplified_with_newline list")
            return np.nan
        
        if len(item) > 512:
            self.continuous_carbon.append(item)
            print(f"Continuous carbon item '{item}' append to too long ")
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
                # sa = sascorer.calculateScore(mol)
                # largest_ring_size = self.get_largest_ring_size(mol)
                # cycle_score = max(largest_ring_size - 6, 0)
                # plogp = logp - sa - cycle_score if logp and sa and largest_ring_size else None
            except Exception as e:
                self.fail2compute_property.append(item)
                print(f"Error item '{item}' appended to fail2compute_property list")
                print(f"Error occurred while computing molecular logP: {e}")
                return np.nan
            
            try:
                fragment_mol = Chem.MolFromSmiles(fragment)
                contain_or_not = mol.HasSubstructMatch(fragment_mol)
                if contain_or_not:
                    self.contain_substructure.append(item)
                    # self.draw_sub_in_full(mol, fragment_mol, len(self.contain_substructure), draw_path)
                else:
                    self.dont_contain_substructure.append(item)
                    return np.nan
            except Exception as e:
                self.fail2processHasSubstructMatch.append(item)
                print(f"Error item '{item}' appended to fail2process HasSubstructureMatch")
                print(f"Error occurred while process HasSubstructMatch: {e}")
                return np.nan

            # Incremental writing to CSV file
            with open(simplified_selfies_smiles_logp_csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([prompt, item, selfies_string, smiles_string, logp])
            # self.success_2_compute_plogp_and_contain_substructure.append(plogp)
            return logp
        return np.nan

    def process_jsonl(self, input_file, output_success_file, simplified_selfies_smiles_logp_csv_file):
        with open(input_file, 'r') as f, \
            open(output_success_file, 'w') as f_success:
            for line in f:
                data = json.loads(line.strip())
                simplified = data.get("Output")
                prompt = data.get("Input")
                # fragment = re.findall(r"'(.*?)'", prompt)[0]
                fragment = "O=CNC1=CC=NN1C"
                if simplified is not None and fragment is not None:
                    success_logp_and_contain_sub = self.function_A(simplified, fragment, simplified_selfies_smiles_logp_csv_file, prompt)
                    if np.isnan(success_logp_and_contain_sub):
                        pass
                    else:
                        data["logp"] = success_logp_and_contain_sub
                        json.dump(data, f_success)
                        f_success.write('\n')

        print("新分子个数:", len(self.novol_inference))
        print("------------------------------------")
        print("非法类型统计：")
        print("simplified_with_three_quotes:", len(self.simplified_with_three_quotes))
        print("simplified_with_newline:", len(self.simplified_with_newline))
        print("simplified_with_too_long_carbon", len(self.continuous_carbon))
        print("selfies_unbalanced:", len(self.selfies_unbalanced))
        print("selfies2smiles_failed:", len(self.selfies2smiles_failed))
        print("smiles_unvalid:", len(self.smiles_unvalid))
        print("fail2compute_property:", len(self.fail2compute_property))
        print("fail2compute_substructure:", len(self.fail2processHasSubstructMatch))
        print("------------------------------------")
        print(f"非法分子个数：{len(self.novol_inference) - len(self.contain_substructure) - len(self.dont_contain_substructure)}")
        print(f"合法分子个数：{len(self.contain_substructure) + len(self.dont_contain_substructure)}")
        print("------------------------------------")
        print("合法且不包含子结构分子个数：", len(self.dont_contain_substructure))
        print("合法且包含子结构分子个数:", len(self.contain_substructure))
        print("合法且包含子结构分子占总推理分子比例:", len(self.contain_substructure) / 46934)
        print("--------------------------------")

    def counter(self, inference_prompt_file, input_csv_file, output_csv_file):
    # def counter(inference_prompt_file, input_csv_file, output_csv_file):
        # # Step 1: Read prompt.txt and extract logp values
        # logp_values = []
        # logp_pattern = re.compile(r'logp:\s*([-+]?\d*\.\d{2})')
        # with open(inference_prompt_file, 'r') as file:
        #     for line in file:
        #         match = logp_pattern.search(line)
        #         if match:
        #             # 提取匹配到的数字
        #             logp_value = float(match.group(1))
        #             logp_values.append(logp_value)
        #         else:
        #             print(f"No match(logp) found in txt line: {line.strip()}")  # 打印没有匹配的行，便于调试
        
        # # Step 2: Read data.csv and extract logp column
        # logp_values2 = []
        # with open(input_csv_file, 'r') as file:
        #     reader = csv.DictReader(file)
        #     for row in reader:
        #         logp_values2.append(row['logp'])

        # # Step 3: Count the number of equal logp values
        # equal_count = sum(1 for logp1, logp2 in zip(logp_values, logp_values2) if logp1 == logp2)
        # total_count = len(logp_values)
        # proportion_equal = equal_count / total_count if total_count > 0 else 0
        # print("--------------------------------")
        # print(f"total inference: {total_count}")
        # print(f"propagation of prompt logp that align with generate logp: {proportion_equal}")
        # print("--------------------------------")
        # # Step 4: Write to a new CSV file
        # with open(output_csv_file, 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['prompt', 'generate'])
        #     for logp1, logp2 in zip(logp_values, logp_values2):
        #         writer.writerow([logp1, logp2])

            # # Write the proportion of equal logp values at the end of the file
            # writer.writerow([])
            # writer.writerow(['Proportion of equal logp values', proportion_equal])

        # 读取CSV文件
        df = pd.read_csv(input_csv_file)
        # 统计logp列的最大值和最小值
        logp_max = df['logp'].max()
        logp_min = df['logp'].min()

        # 统计频次
        logp_count = df['logp'].value_counts()
        print("--------------------------------")
        print(f"logp最大值: {logp_max}")
        print(f"logp最小值: {logp_min}")
        print("频次:")
        print(logp_count)
        print("--------------------------------")

    def similarity_calculation(self, idx1):
        fp1 = self.df.at[idx1, 'mfp2']
        simlist = [DataStructs.DiceSimilarity(fp1, self.df.at[idx2, 'mfp2']) for idx2 in self.df.index]
        return sum(simlist)

    def compute_similarity(self, simplified_selfies_smiles_logp_csv_file):
        df1 = pd.read_csv(simplified_selfies_smiles_logp_csv_file)
        # 在计算之前，检查smiles列中的数据类型
        # 保留smiles列中不是float类型的数据
        df2 = df1[~df1['smiles'].apply(lambda x: isinstance(x, float))]
        self.df = df2
        # self.logp_in_range(self.df)
        # self.logp_in_range(logp_inrange_csv)

        # start_time1 = time.time()
        PandasTools.AddMoleculeColumnToFrame(self.df, 'smiles', 'mol', includeFingerprints=True)
        # end_time1 = time.time()
        # print("Time taken to add molecules to dataframe:", end_time1 - start_time1, "seconds")

        # start_time2 = time.time()
        fplist = [Chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in self.df['mol']]  ## 2的意思是r = 2 半径
        self.df['mfp2'] = fplist
        # end_time2 = time.time()
        # print("Time taken to compute Morgan fingerprints and add to dataframe:", end_time2 - start_time2, "seconds")

        # start_time3 = time.time()
        with Pool() as pool:
            # similarity_sum = sum(pool.map(self.similarity_calculation_wrapper, df.index)) ## map（函数，可迭代器）
            similarity_sum = sum(pool.map(self.similarity_calculation, self.df.index)) ## map（函数，可迭代器）
        similarity_average = (similarity_sum - self.df.shape[0]) / ((self.df.shape[0] ** 2) - self.df.shape[0])
        print("--------------------------------")
        print(f"符合条件分子相似度为：{similarity_average}")
        print(f"符合条件分子多样性为: {1.0 - similarity_average}")
        # end_time3 = time.time()
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
    if line_count != 46934:
        print("扑街啊你个二五仔，不是46934条啦！")
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



















# 设置日志文件路径
def setup_logging(input_file_path):
    base, _ = os.path.splitext(input_file_path)
    log_filename = f"{base}_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return log_filename

# 处理 JSONL 文件和计算指标的函数
def analyze_data(input_file_path):
    # 自动生成输出文件路径
    base, _ = os.path.splitext(input_file_path)
    output_csv_path = base + '_compare.csv'

    # 设置日志记录
    log_filename = setup_logging(input_file_path)

    # 初始化两个列表
    list1 = []
    list2 = []

    # 读取 JSONL 文件
    with open(input_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            
            # 从 Input 字段中提取 logp 值
            input_text = data['Input']
            input_logp_str = input_text.split('logp: ')[1].split(',')[0]
            input_logp = float(input_logp_str)
            list1.append(input_logp)
            
            # 从 Output 字段中的 logp 值提取
            output_logp = data['logp']
            list2.append(output_logp)

    # 保存到 CSV 文件
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Input_logp', 'Output_logp'])
        for input_logp, output_logp in zip(list1, list2):
            writer.writerow([input_logp, output_logp])

    # 计算皮尔逊相关性系数
    correlation, _ = pearsonr(list1, list2)
    logging.info(f'皮尔逊相关性系数: {correlation}')

    # 读取 CSV 文件
    df = pd.read_csv(output_csv_path)

    try:
        y_true = df['Input_logp'].values
        y_pred = df['Output_logp'].values
    except KeyError as e:
        logging.error(f"错误：列名 {e} 不存在。请检查 CSV 文件的列名。")
        raise

    # 计算均方误差（MSE）
    mse = mean_squared_error(y_true, y_pred)
    logging.info(f"Mean Squared Error (MSE): {mse}")

    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)
    logging.info(f"Root Mean Squared Error (RMSE): {rmse}")

    # 计算平均绝对误差（MAE）
    mae = mean_absolute_error(y_true, y_pred)
    logging.info(f"Mean Absolute Error (MAE): {mae}")

    # 计算决定系数（R^2）
    r2 = r2_score(y_true, y_pred)
    logging.info(f"R-squared (R^2): {r2}")

    # 计算平均绝对百分比误差（MAPE），处理基准值为0的情况
    non_zero_indices = y_true != 0
    y_true_non_zero = y_true[non_zero_indices]
    y_pred_non_zero = y_pred[non_zero_indices]

    if len(y_true_non_zero) > 0:
        mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100
    else:
        mape = float('inf')  # 如果没有非零基准值，则 MAPE 为无穷大

    logging.info(f"Mean Absolute Percentage Error (MAPE): {mape}%")

    logging.info(f"Logs and results saved to {log_filename}")



















def main():
    parser = argparse.ArgumentParser(description='Inference process script')
    parser.add_argument('--base_workspace', type=str, required=True, help='Base workspace directory')
    parser.add_argument('--training_jsonl_file', type=str, required=True, help='Path to training JSONL file')
    parser.add_argument('--prompt_file', type=str, required=True, help='Path to inference prompt file')

    args = parser.parse_args()

    base_workspace = args.base_workspace
    training_jsonl_file = args.training_jsonl_file
    prompt_file = args.prompt_file
    
    output_save_path = f'{base_workspace}/process_mine'
    inference_file = f'{base_workspace}/prediction_result.jsonl'
    noval_inference_file = f'{output_save_path}/novol_inference_output.jsonl'
    output_success_file = f'{output_save_path}/successfully_computed.jsonl'
    # output_fail_file = f'{output_save_path}/fail_to_compute.jsonl'
    simplified_selfies_smiles_logp_csv_file = f'{output_save_path}/valid_simplified_selfies_smiles_logp.csv'
    prompt_generate_file = f'{output_save_path}/prompt_generate.csv'

    quchong(training_jsonl_file, inference_file, noval_inference_file)

    with open(simplified_selfies_smiles_logp_csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['input_prompt', 'simplified', 'selfies', 'smiles', 'logp'])

    processor = DataProcessor()
    processor.process_jsonl(noval_inference_file, output_success_file, simplified_selfies_smiles_logp_csv_file)
    
    processor.counter(prompt_file, simplified_selfies_smiles_logp_csv_file, prompt_generate_file)
    processor.compute_similarity(simplified_selfies_smiles_logp_csv_file)



    # 进行分析
    analyze_data(output_success_file)


if __name__ == "__main__":
    main()
