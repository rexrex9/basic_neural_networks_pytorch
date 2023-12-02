from os.path import join as osp
import os
import datetime

REPORTS_DIR = osp(os.path.split(os.path.realpath(__file__))[0],'reports')

DETAILS_NAME = 'details.json'
DETAILS_CSV = 'detail_scores.csv'
OVERALL_NAME = 'overall.txt'

def __gen_report_dir_path(custom_prefix):
    today = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    name = f'{custom_prefix}{today}'
    p = osp(REPORTS_DIR,name)
    if not os.path.exists(p):
        os.mkdir(p)
    return p

def __gen_details_report(result,report_dir_path):
    df = result.to_pandas()
    df.to_json(osp(report_dir_path,DETAILS_NAME),orient='records',force_ascii=False,indent=4)
    df.drop('answer',axis=1,inplace=True)
    df.drop('contexts',axis=1,inplace=True)
    df.drop('ground_truths',axis=1,inplace=True)
    df.to_csv(osp(report_dir_path,DETAILS_CSV),index=True,encoding='utf-8-sig')

def __gen_overall_report(result,report_dir_path):
    scores = result.copy()
    score_strs = [f"{k}: {v:0.4f}" for k, v in scores.items()]
    s = "\n".join(score_strs)
    with open(osp(report_dir_path,OVERALL_NAME),'w') as f:
        f.write(s)

def gen_report(result,custom_prefix=''):
    report_dir_path = __gen_report_dir_path(custom_prefix)
    __gen_details_report(result,report_dir_path)
    __gen_overall_report(result,report_dir_path)
    return report_dir_path


