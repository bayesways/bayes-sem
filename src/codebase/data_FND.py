import pandas as pd
import numpy as np


def get_FND_data(return_dict=True, alt_order=False):
    df = pd.read_csv("../dat/real_data/FND.csv")

    df_dscr = pd.read_excel(
        "../dat/real_data/FTND Description.xlsx", skiprows=4)

    question_field_names = df_dscr.iloc[np.r_[6:13], 1].values

    dfq = df.loc[:, question_field_names]
    dfq = dfq[dfq['FNSMOKE'] == 1]
    dfq = dfq.iloc[:, 1:].dropna().reset_index(drop=True).astype(int)
    dfq['FNFIRST'] = (dfq.FNFIRST > 2).astype(int)
    dfq['FNNODAY'] = (dfq.FNNODAY > 1).astype(int)

    listofquestions = ['FNFIRST', 'FNGIVEUP',
                       'FNFREQ', 'FNNODAY', 'FNFORBDN', 'FNSICK']
    listofquestions2 = ['FNGIVEUP',
                        'FNFREQ', 'FNNODAY', 'FNFORBDN', 'FNSICK', 'FNFIRST']

    dfq = dfq.loc[:, listofquestions]

    if alt_order:
        dfq = dfq.loc[:, listofquestions2]

    if return_dict:
        data = dict()
        data['D'] = dfq.values
        data['N'], data['J'] = dfq.shape
        data['K'] = 2

        return data
    else:
        return dfq
