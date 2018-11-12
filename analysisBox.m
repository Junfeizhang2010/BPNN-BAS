function analysisBox(data, gap, xLabel)
    %% 各模型预测差值箱型图分析
    % 输入：
    %   gap     - 各模型预测值与真实值的差（每一列是一个箱）
    
    gap(:, all(gap == 0, 1)) = [];      % 去掉全0的列
    gap(:, all(gap == inf, 1)) = [];	% 去掉全inf的列
    gap(:, all(gap == -inf, 1)) = [];	% 去掉全-inf的列
    figurePara = setFigurePara(data.dataIndex, data.posOutput);
    figurePara.label.title = 'Gap of Predicted and Experiment Values of Each Model';
    figurePara.fold = ['savedFigure\', data.dataIndex];
    figurePara.preserve.name = '各模型预测差值箱型图';
    figurePara.label.y = 'Gap of predicted and experiment values';
    figurePara.label.x = xLabel;
    drawBox(gap, figurePara);