#import train_test_data_prepare as sdp
from train_test_data_prepare import get_xy_data
from predict_sandhi_window_bilstm import train_predict_sandhi_window
from split_sandhi_window_seq2seq_bilstm import train_sandhi_split
from sklearn.model_selection import train_test_split

inwordlen = 5

dl = get_xy_data("../Data/sandhiset.txt")

# Split the training and testing data
dtrain, dtest = train_test_split(dl, test_size=0.2, random_state=1)

#predict the sandhi window
sl = train_predict_sandhi_window(dtrain, dtest, 1)

if len(sl) == len(dtest):
    for i in range(len(dtest)):
        start = sl[i]
        end = sl[i] + inwordlen
        flen = len(dtest[i][3])
        if end > flen:
            end = flen
        dtest[i][2] = dtest[i][3][start:end]
        dtest[i][4] = start
        dtest[i][5] = end
else:
    print("error")

#split the sandhi
results = train_sandhi_split(dtrain, dtest, 1)

if len(results) == len(dtest):
    passed = 0
    failed = 0

    for i in range(len(dtest)):
        start = dtest[i][4]
        end = dtest[i][5]
        splitword = dtest[i][3][:start] + results[i] + dtest[i][3][end:]
        actword = dtest[i][6] + '+' + dtest[i][7]
        if splitword == actword:
            passed = passed + 1
        else:
            failed = failed + 1
    print(passed)
    print(failed)
    print(passed*100/(passed+failed))
else:
    print("error")
