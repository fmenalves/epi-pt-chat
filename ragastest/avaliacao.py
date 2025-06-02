
import pandas as pd


from ragastest.agent_testes import agent_process_query

from ragastest.support_testes import process_query

from ragastest.ragasscore import ragas_score


###################  Esboço

per_res = pd.read_csv("ragastest/Perguntas e resposta - Cópia de Folha1.csv", delimiter=",")

retriever_list = [5, 7, 10, 12 , 15, 20, 25, 30]
rerank_list = [2, 3, 4, 5, 7, 10, 12, 13, 15]

produtos = ["Rinvoq", "Ozempic", "Paclitaxel Accord MG", "Diovan", "Influvac Tetra", "Comirnaty JN.1", "Triticum", "Depakine", "Lenalidomida Tecnigen MG", "Ciplox MG"]

dosagem = ["15 mg", "0.5 mg/0.37 ml", "6 mg/ml", "80 mg", "Associação", "30 µg/0.3 ml", "100 mg", "40 mg/ml", "10 mg", "500 mg"]

for i in range(len(per_res)):
    for ret in retriever_list:
        for rer in (r for r in rerank_list if r < ret):
            result = agent_process_query(
                query = per_res["Pergunta"][i],
                products = produtos[i // 5],
                strength = dosagem[i // 5],
                enhance_query = False,
                ret_similarity_top_k = ret, 
                rer_top_n = rer
            )
            
            rag_score = ragas_score(
                Pergunta = per_res["Pergunta"][i],
                Resposta = result['response'],
                contextos = result['contexts'],
                ground_truth = per_res["Resposta"][i]
            )

            time = result['execution_time']
                                                     




