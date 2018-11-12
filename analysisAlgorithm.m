function [err, R, gap, bestIteration] = analysisAlgorithm(data, tuneMode, algoMode, opt)
    %% �㷨�ȽϷ���
    % ���룺
    %   data.all                    - �������ݣ�ÿһ��һ�����������ֵ����β��
    %       .normalizedAll          - ��һ�������ݣ�ÿһ��һ�����������ֵ����β��
    %       .count                  - ���ݸ���
    %       .dimCount               - ����ά��
    %       .dataIndex              - ���ݼ�����
    %       .posY                   - ��һ��Ŀ��ֵ�����к�
    %       .trainPer               - ѵ��������
    %       .isColumnAsSimple       - �Ƿ���ÿ��Ϊһ����������ʽ����
    %   	.trainXAllColumn        - ѵ��������ֵ��ÿһ��һ��������
    %   	.trainYAllColumn        - ѵ����Ŀ��ֵ��ֻ��һ��Ŀ��ֵ����������
    %   	.testXAllColumn         - ���Լ�����ֵ��ÿһ��һ��������
    %   	.testYAllColumn         - ���Լ�Ŀ��ֵ��ֻ��һ��Ŀ��ֵ����������
    %   	.trainXAllRow           - ѵ��������ֵ��ÿһ��һ��������
    %   	.trainYAllRow           - ѵ����Ŀ��ֵ��ֻ��һ��Ŀ��ֵ����������
    %   	.testXAllRow            - ���Լ�����ֵ��ÿһ��һ��������
    %   	.testYAllRow            - ���Լ�Ŀ��ֵ��ֻ��һ��Ŀ��ֵ����������
    %       .crossColumn{i}.trainX  - һ��������֤��ѵ��������ֵ��ÿһ��һ��������
    %                      .trainY  - һ��������֤��ѵ����Ŀ��ֵ��ֻ��һ��Ŀ��ֵ����������
    %                      .testX   - һ��������֤�Ĳ��Լ�����ֵ��ÿһ��һ��������
    %                      .testY   - һ��������֤�Ĳ��Լ�Ŀ��ֵ��ֻ��һ��Ŀ��ֵ����������
    %       .crossRow{i}.trainX     - һ��������֤��ѵ��������ֵ��ÿһ��һ��������
    %                   .trainY     - һ��������֤��ѵ����Ŀ��ֵ��ֻ��һ��Ŀ��ֵ����������
    %                   .testX      - һ��������֤�Ĳ��Լ�����ֵ��ÿһ��һ��������
    %                   .testY      - һ��������֤�Ĳ��Լ�Ŀ��ֵ��ֻ��һ��Ŀ��ֵ����������
    %   tuneMode                    - �Ż��㷨
    %   algoMode                    - ģ���㷨
    %   opt                         - �㷨����
    % �����
    %   err                         - ������֤ƽ�����
    %   R                           - ������֤ƽ�����ϵ��
    %   gap                         - Ԥ��ֵ����ʵֵ�Ĳ��������
    %   bestIteration               - ��������ͼ����
    
    %% �Ż�ȷ��ģ�Ͳ���
    [para, bestIteration] = tune(data, tuneMode, algoMode, opt);
    
    %% ������֤����ģ��
    [err, R, gap, model] = evaluate(data, para, tuneMode, algoMode, []);
    mkdir(getCurrentPath(mfilename('fullpath')), ['savedData\' data.dataIndex]);
    save([getCurrentPath(mfilename('fullpath')), 'savedData\' data.dataIndex '\ʹ��', tuneMode, '�㷨�Ż�', algoMode, 'ģ�͵�����'], 'para', 'err', 'R', 'gap', 'model');
    
    %% �Ա�δ�Ż����
    untunedCompare(data, err, R, tuneMode, algoMode, opt);
    
    %% ָ�������Է���
    analysisSensitivity(data, model, tuneMode, algoMode, []);