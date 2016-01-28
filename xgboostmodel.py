import xgboost as xgb
import csv
from baselinemodel import read_data


# get training and test data
train_data, test_data = read_data('./data/train.csv', './data/test.csv')
# training input and corresponding output label
train_x, train_y = train_data[1:, :-1], train_data[1:, -1:]

# fit xgboost classifier
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_x, train_y.ravel())
pred_y = gbm.predict(test_data[1:, :])

y_pred1 = [int(predict) for predict in pred_y]
id = [int(test_p) for test_p in test_data[:,0]]
#print("Writing to file")
with open('prediction.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows([['Id','Response']])
    writer.writerows(zip(id,y_pred1))
f.close()
print("Ban gaya Data Scientist!!!")
