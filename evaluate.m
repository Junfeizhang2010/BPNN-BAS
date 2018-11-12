function [err, R, gap, model] = evaluate(data, para, tuneMode, algoMode, opt)
    %% 测试BP神经网络
    % 输入：
    %   data                - 数据集
    %   para                - 列向量，优化算法推荐的BP神经网络权重和偏置
    %   algoMode            - 模型算法
    %   isDraw              - 是否画图
    % 输出：
    %   err                 - 十折交叉验证的测试集误差均值
    %   R                   - 十折交叉验证的测试集相关系数均值
    %   gap                 - 预测值与真实值的差（行向量）
    %   model               - 最终预测时用的模型
    
    crossColumn = data.crossColumn;
    %% 训练与测试
    disp(['            ●正在进行' algoMode '模型评估...']);
    crossTime = size(crossColumn, 2);
    predicted = [];
    experimental = [];
    for j = 1 : crossTime
        switch algoMode
            case 'BP'
                [err(j), R(j), predictedTemp, model] = getErrorBP(data, j, para, false); % 使用十折交叉验证后，只能计算每次的测试集误差
            otherwise
                [err(j), R(j), predictedTemp, model] = getErrorNormal(data, j, para, algoMode); 
        end
        predicted = [predicted, predictedTemp];                 % 行向量
        experimental = [experimental, crossColumn{j}.testY];    % 行向量
    end
%     err = mean(err);
%     R = mean(R);
    err = mse(predicted, experimental);
    R = min(min(corrcoef(predicted, experimental)));
    gap = abs(predicted - experimental);
    disp(['              ' algoMode '模型评估完毕。']);
    
    %% 画图
    if ~isempty(opt) || ~isfield(opt, 'show') || ~isfield(opt.show, 'isDraw') || opt.show.isDraw
        % 二维回归图（测试集）
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.title = [algoMode 'Regression of TestSet with Cross'];
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.preserve.name = ['使用' tuneMode '算法优化的' algoMode '模型十折交叉验证测试集回归图'];
        drawRegression(experimental, predicted, figurePara);
        
        % 酷炫预测图（测试集）
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.title = [algoMode 'Predictive of TestSet with Cross'];
        figurePara.label.x = 'Data Number';
        figurePara.label.y2 = 'Residual';
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.preserve.name = ['使用' tuneMode '算法优化的' algoMode '模型十折交叉验证测试集预测图'];
        figurePara.isSingle = false;
        drawPredicted(experimental, predicted, figurePara);
    end