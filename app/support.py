from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
)
from llama_index.core import PromptTemplate


text_qa_template_str = (
    "Context information is"
    " below.\n---------------------\n{context_str}\n---------------------\nUsing"
    " the context information and not prior knowledge, answer"
    " the question: {query_str}\nIf the context isn't helpful, you can also"
    " answer the question on your own."
    "Respond always in the language portuguese from portugal and do not use any word that is specific from brazil."
    "Cite the source of the new information if you use it.\n"
    "Answer:\n"
)
text_qa_template = PromptTemplate(text_qa_template_str)

refine_template_str = (
    "The original question is as follows: {query_str}\nWe have provided an"
    " existing answer: {existing_answer}\nWe have the opportunity to refine"
    " the existing answer (only if needed) with some more context"
    " below.\n------------\n{context_msg}\n------------\nUsing the"
    " context information and not prior knowledge, update or repeat the existing answer.\n"
    " Respond always in the language portuguese from portugal and do not use any word that is specific from brazil.\n"
    "Cite the source of the new information if you use it.\n"
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)
refine_template = PromptTemplate(refine_template_str)


# print(list(map(' '.join, zip(words[:-1], words[1:]))))
def create_filters(query, metadatasource):
    f = []
    black_list = ["comprimido", "cápsula"]
    words = query.replace("?", "").split()
    for w2 in list(map(" ".join, zip(words[:-1], words[1:]))):
        if metadatasource["Subs"].str.contains(w2, case=False).any():
            # print("found")
            print(w2, "subs2")

            f.append(MetadataFilter(key="substancia", value=w2, operator="in"))
            break
    for w in words:
        if (
            len(w) > 5
            and metadatasource["Subs"].str.contains(w, case=False).any()
            and w not in black_list
        ):
            # print("found")
            print(w, "subs")
            f.append(MetadataFilter(key="substancia", value=w, operator="in"))
            break
    for w2 in list(map(" ".join, zip(words[:-1], words[1:]))):
        if metadatasource["Nome Comercial"].str.contains(w2, case=False).any():
            # print("found")
            print(w2, "comer")
            f.append(MetadataFilter(key="Nome_Comercial", value=w2, operator="in"))
            break
    for w in words:
        if (
            len(w) > 5
            and metadatasource["Nome Comercial"].str.contains(w, case=False).any()
            and w not in black_list
        ):
            # print("found")
            print(w, "Nome Comercial")
            f.append(MetadataFilter(key="Nome_Comercial", value=w, operator="in"))
            break
    return MetadataFilters(filters=f, condition="or")


def get_metadata(metadatasource, nfilename, file_path):
    extra_metadata = metadatasource[metadatasource["NewFilename"] == nfilename].values

    if len(extra_metadata) == 0:
        nfilename = file_path.split("/")[-1][:-4]
        extra_metadata = metadatasource[
            metadatasource["NewFilename"] == nfilename
        ].values

    if len(extra_metadata) == 0:
        print("No metadata found for", nfilename)
        extra_metadata = [["", "", "", "", "", ""]]
    ff = {
        "Nome_Comercial": extra_metadata[0][1],
        "substancia": extra_metadata[0][2],
        "Forma_Farmaceutica": extra_metadata[0][3],
        "Dosagem": extra_metadata[0][4],
        "Titular_AIM": extra_metadata[0][5],
    }

    return ff
