function analysisBox(data, gap, xLabel)
    %% ��ģ��Ԥ���ֵ����ͼ����
    % ���룺
    %   gap     - ��ģ��Ԥ��ֵ����ʵֵ�Ĳÿһ����һ���䣩
    
    gap(:, all(gap == 0, 1)) = [];      % ȥ��ȫ0����
    gap(:, all(gap == inf, 1)) = [];	% ȥ��ȫinf����
    gap(:, all(gap == -inf, 1)) = [];	% ȥ��ȫ-inf����
    figurePara = setFigurePara(data.dataIndex, data.posOutput);
    figurePara.label.title = 'Gap of Predicted and Experiment Values of Each Model';
    figurePara.fold = ['savedFigure\', data.dataIndex];
    figurePara.preserve.name = '��ģ��Ԥ���ֵ����ͼ';
    figurePara.label.y = 'Gap of predicted and experiment values';
    figurePara.label.x = xLabel;
    drawBox(gap, figurePara);