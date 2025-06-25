
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


results = [] 


for i in range(len(per_res)):

    pergunta = per_res["Pergunta"][i]
    ground_truth = per_res["Resposta"][i]
    produto  = produtos[i // 5]
    dose     = dosagem[i // 5]

    for ret in retriever_list:
        for rer in (r for r in rerank_list if r < ret):

            ag_res = agent_process_query(
                query = pergunta,
                products = produto,
                strength = dose,
                enhance_query = False,
                ret_similarity_top_k = ret, 
                rer_top_n = rer,
                Cohere = False
            )

            ag_rag_score = ragas_score(
                pergunta = pergunta,
                resposta = ag_res['response'],
                contextos = ag_res['contexts'],
                ground_truth = ground_truth
            )

            for _, row in ag_rag_score.iterrows():

                results.append(
                    dict(
                        sistema="agent",
                        pergunta=pergunta,
                        ret=ret,
                        rer=rer,
                        metrica=row["Métrica"],
                        valor=row["Resultado"],
                        exec_time=ag_res["execution_time"],
                    )
                )



            base_res = process_query(
                query = pergunta,
                products = produto,
                strength = dose,
                enhance_query = False,
                ret_similarity_top_k = ret, 
                rer_top_n = rer
            )                       

            base_rag_score = ragas_score(
                pergunta = pergunta,
                resposta = base_res['response'],
                contextos = base_res['contexts'],
                ground_truth = ground_truth
            )

            for _, row in base_rag_score.iterrows():

                results.append(
                    dict(
                        sistema="base",
                        pergunta=pergunta,
                        ret=ret,
                        rer=rer,
                        metrica=row["Métrica"],
                        valor=row["Resultado"],
                        exec_time=base_res["execution_time"],
                    )
                )
                                                     



df_all = pd.DataFrame(results)
