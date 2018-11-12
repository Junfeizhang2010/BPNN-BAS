function analysisTune(data, bestIteration, algoMode)
    %% ��ģ���Ż����̵�������
    % ���룺
    %   bestIteration     - ��ģ���Ż����̵������ݣ�ÿ��һ���㷨���Ż����̵������ݣ�
    
    % ȥ��С��������
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
    figurePara.preserve.name = 'BAS�㷨�Ż���ģ�Ͳ����ĵ�����������ͼ';
    figurePara.label.y = figurePara.label.mse;
    figurePara.label.x = 'Iteration';
    figurePara.label.legend = algoMode;
    drawMultipleLine(bestIteration, figurePara);