function analysisPartialDependency(data)
    %% ������������
    % ���룺
    %   data    - ���ݼ�
    
    figurePara = setFigurePara(data.dataIndex, data.posOutput);
    for i = 1 : data.inputDimCount
        figurePara.label.title = 'Partial dependency graph';
        figurePara.fold = ['savedFigure\' data.dataIndex];
        figurePara.preserve.name = [data.dataIndex '��������ͼ'];
        figurePara.label.x = figurePara.label.input{i};
        drawSinglePoints(data.all(:,i), data.all(:,end), figurePara);
    end