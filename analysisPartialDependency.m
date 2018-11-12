function analysisPartialDependency(data)
    %% 部分依赖分析
    % 输入：
    %   data    - 数据集
    
    figurePara = setFigurePara(data.dataIndex, data.posOutput);
    for i = 1 : data.inputDimCount
        figurePara.label.title = 'Partial dependency graph';
        figurePara.fold = ['savedFigure\' data.dataIndex];
        figurePara.preserve.name = [data.dataIndex '部分依赖图'];
        figurePara.label.x = figurePara.label.input{i};
        drawSinglePoints(data.all(:,i), data.all(:,end), figurePara);
    end