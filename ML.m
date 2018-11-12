clear;close all;
% '水泥材料'：    Coal-crete_dataset
% '二维电子气'：  Hall parameters of GaN
% 'Cr浓度'：      Chromate sensor response predsiction
% '化学材料'：    Coal-crete_dataset 2_different grouting material
% '葡萄糖'：      biosensor_data
% '漂移补偿'：    biosensor_data_compensation

algoMode = {'BP',           'SVM',       'DT',      'RF',       'KNN',      'LR',       'NB',        'AD'};
            % BP神经网络，   支持向量机，  决策树，    随机森林，   K近邻，    逻辑回归，   朴素贝叶斯，  adaboost
tuneMode = {'BAS'};
            % 天牛须
            
opt1.data.dataName = '水泥材料';
opt1.data.algoMode = algoMode(3);
opt1.data.tuneMode = tuneMode(1);
experiment(opt1);

opt2.data.dataName = '化学材料';
opt2.data.algoMode = algoMode(1:6);
opt2.data.tuneMode = tuneMode(1);
% experiment(opt2);

opt3.data.dataName = '吸波混凝土';
opt3.data.algoMode = algoMode(1:6);
opt3.data.tuneMode = tuneMode(1);
opt3.data.crossNum = 5;
opt3.tune.time = 1;
opt3.BP.hLMaxCount = 3;
opt3.BP.BAS.n = 30;
opt3.compare.pointCount = 30;
% experiment(opt3);