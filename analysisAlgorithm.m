function [err, R, gap, bestIteration] = analysisAlgorithm(data, tuneMode, algoMode, opt)
    %% 算法比较分析
    % 输入：
    %   data.all                    - 所有数据（每一行一个样本，输出值在行尾）
    %       .normalizedAll          - 归一化的数据（每一行一个样本，输出值在行尾）
    %       .count                  - 数据个数
    %       .dimCount               - 数据维度
    %       .dataIndex              - 数据集名称
    %       .posY                   - 第一个目标值所在列号
    %       .trainPer               - 训练集比例
    %       .isColumnAsSimple       - 是否以每列为一个样本的形式呈现
    %   	.trainXAllColumn        - 训练集特征值（每一列一个样本）
    %   	.trainYAllColumn        - 训练集目标值（只有一个目标值，行向量）
    %   	.testXAllColumn         - 测试集特征值（每一列一个样本）
    %   	.testYAllColumn         - 测试集目标值（只有一个目标值，行向量）
    %   	.trainXAllRow           - 训练集特征值（每一行一个样本）
    %   	.trainYAllRow           - 训练集目标值（只有一个目标值，列向量）
    %   	.testXAllRow            - 测试集特征值（每一行一个样本）
    %   	.testYAllRow            - 测试集目标值（只有一个目标值，列向量）
    %       .crossColumn{i}.trainX  - 一个交叉验证的训练集特征值（每一列一个样本）
    %                      .trainY  - 一个交叉验证的训练集目标值（只有一个目标值，行向量）
    %                      .testX   - 一个交叉验证的测试集特征值（每一列一个样本）
    %                      .testY   - 一个交叉验证的测试集目标值（只有一个目标值，行向量）
    %       .crossRow{i}.trainX     - 一个交叉验证的训练集特征值（每一行一个样本）
    %                   .trainY     - 一个交叉验证的训练集目标值（只有一个目标值，列向量）
    %                   .testX      - 一个交叉验证的测试集特征值（每一行一个样本）
    %                   .testY      - 一个交叉验证的测试集目标值（只有一个目标值，列向量）
    %   tuneMode                    - 优化算法
    %   algoMode                    - 模型算法
    %   opt                         - 算法参数
    % 输出：
    %   err                         - 交叉验证平均误差
    %   R                           - 交叉验证平均相关系数
    %   gap                         - 预测值与真实值的差（行向量）
    %   bestIteration               - 迭代收敛图数据
    
    %% 优化确定模型参数
    [para, bestIteration] = tune(data, tuneMode, algoMode, opt);
    
    %% 交叉验证评估模型
    [err, R, gap, model] = evaluate(data, para, tuneMode, algoMode, []);
    mkdir(getCurrentPath(mfilename('fullpath')), ['savedData\' data.dataIndex]);
    save([getCurrentPath(mfilename('fullpath')), 'savedData\' data.dataIndex '\使用', tuneMode, '算法优化', algoMode, '模型的数据'], 'para', 'err', 'R', 'gap', 'model');
    
    %% 对比未优化情况
    untunedCompare(data, err, R, tuneMode, algoMode, opt);
    
    %% 指标敏感性分析
    analysisSensitivity(data, model, tuneMode, algoMode, []);