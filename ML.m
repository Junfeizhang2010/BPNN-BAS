clear;close all;
% 'ˮ�����'��    Coal-crete_dataset
% '��ά������'��  Hall parameters of GaN
% 'CrŨ��'��      Chromate sensor response predsiction
% '��ѧ����'��    Coal-crete_dataset 2_different grouting material
% '������'��      biosensor_data
% 'Ư�Ʋ���'��    biosensor_data_compensation

algoMode = {'BP',           'SVM',       'DT',      'RF',       'KNN',      'LR',       'NB',        'AD'};
            % BP�����磬   ֧����������  ��������    ���ɭ�֣�   K���ڣ�    �߼��ع飬   ���ر�Ҷ˹��  adaboost
tuneMode = {'BAS'};
            % ��ţ��
            
opt1.data.dataName = 'ˮ�����';
opt1.data.algoMode = algoMode(3);
opt1.data.tuneMode = tuneMode(1);
experiment(opt1);

opt2.data.dataName = '��ѧ����';
opt2.data.algoMode = algoMode(1:6);
opt2.data.tuneMode = tuneMode(1);
% experiment(opt2);

opt3.data.dataName = '����������';
opt3.data.algoMode = algoMode(1:6);
opt3.data.tuneMode = tuneMode(1);
opt3.data.crossNum = 5;
opt3.tune.time = 1;
opt3.BP.hLMaxCount = 3;
opt3.BP.BAS.n = 30;
opt3.compare.pointCount = 30;
% experiment(opt3);