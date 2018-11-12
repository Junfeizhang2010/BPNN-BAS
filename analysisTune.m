function analysisTune(data, bestIteration, algoMode)
    %% 各模型优化过程迭代分析
    % 输入：
    %   bestIteration     - 各模型优化过程迭代数据（每行一个算法的优化过程迭代数据）
    
    % 去最小迭代次数
    n = length(bestIteration);
    len = length(bestIteration{1});
    for i = 2 : n
        if length(bestIteration{i}) < len, len = length(bestIteration{i}); end
    end
    for i = 1 : n
        iterData(i,:) = length(bestIteration{i});
    end
    figurePara = setFigurePara(data.dataIndex, data.posOutput);
    figurePara.label.title = 'Iterative Convergence of Parameters Tuning with BAS of Each Model';
    figurePara.fold = ['savedFigure\', data.dataIndex];
    figurePara.preserve.name = 'BAS算法优化各模型参数的迭代收敛迭代图';
    figurePara.label.y = figurePara.label.mse;
    figurePara.label.x = 'Iteration';
    figurePara.label.legend = algoMode;
    drawMultipleLine(bestIteration, figurePara);