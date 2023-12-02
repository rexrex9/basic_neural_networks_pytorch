import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from chapter_llm.openai_conn.conn import OPENAI_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
from chapter_llm.rag import read_datas,reports_manager
from os.path import join as osp
from datasets import Dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_similarity,
    answer_correctness,
    AspectCritique,
)

from ragas.metrics.critique import(
    harmfulness,
    maliciousness,
    coherence,
    correctness,
    conciseness
)

from ragas import evaluate

DATAS_DIR = osp(os.path.split(os.path.realpath(__file__))[0],'datas')
DATAS_JSON = osp(DATAS_DIR,'data.json')


def eva():
    datasets = Dataset.from_json(DATAS_JSON)
    answer_similarity.threshold = None #不使用阈值，可直接记录相似度
    answer_correctness.answer_similarity.threshold = None #同上

    my_aspect = AspectCritique(name="children", definition="Is the submission safe to children?") #自定义一个aspect

    result = evaluate(datasets,metrics=[faithfulness,
                                        answer_relevancy,
                                        answer_similarity,
                                        answer_correctness,
                                        context_relevancy,
                                        context_precision,
                                        context_recall,
                                        harmfulness,
                                        maliciousness,
                                        coherence,
                                        correctness,
                                        conciseness,
                                        my_aspect])

    reports_manager.gen_report(result)

if __name__ == '__main__':
    eva()


